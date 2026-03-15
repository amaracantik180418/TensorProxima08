/*
 * TensorProxima08 - AI Training Software Bot
 * Single-file implementation: run registry, epochs, checkpoints, optimizers, loss and metrics.
 */

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.time.*;
import java.util.function.*;
import java.util.stream.*;
import java.util.regex.*;

// -----------------------------------------------------------------------------
// EXCEPTIONS
// -----------------------------------------------------------------------------

class TP08RunNotFoundException extends RuntimeException {
    public TP08RunNotFoundException(String runId) { super("Run not found: " + runId); }
}

class TP08EpochIndexException extends RuntimeException {
    public TP08EpochIndexException(int idx, int max) { super("Epoch index " + idx + " out of range [0," + max + ")"); }
}

class TP08CheckpointException extends RuntimeException {
    public TP08CheckpointException(String msg) { super(msg); }
}

class TP08GradientExplosionException extends RuntimeException {
    public TP08GradientExplosionException(double norm) { super("Gradient norm too large: " + norm); }
}

class TP08ConfigValidationException extends RuntimeException {
    public TP08ConfigValidationException(String field) { super("Invalid config: " + field); }
}

class TP08DatasetEmptyException extends RuntimeException {
    public TP08DatasetEmptyException() { super("Dataset is empty"); }
}

// -----------------------------------------------------------------------------
// CONFIG & CONSTANTS
// -----------------------------------------------------------------------------

final class TP08Constants {
    static final int MAX_EPOCHS_DEFAULT = 100;
    static final int BATCH_SIZE_DEFAULT = 32;
    static final double LEARNING_RATE_DEFAULT = 1e-3;
    static final double GRADIENT_CLIP_NORM = 5.0;
    static final int CHECKPOINT_EVERY_EPOCHS = 5;
    static final String RUN_ID_PREFIX = "tp08_";
    static final int LOSS_SCALE_FACTOR = 1_000_000_000;
    static final long RANDOM_SEED_BASE = 0x8f4B2c1EL;
    private TP08Constants() {}
}

// -----------------------------------------------------------------------------
// TRAINING RUN
// -----------------------------------------------------------------------------

final class TrainingRunRecord {
    private final String runId;
    private final String submitterId;
    private final int epochCount;
    private final byte[] configHash;
    private final long registeredAtEpochMillis;
    private volatile boolean archived;
    private final AtomicInteger epochsRecorded = new AtomicInteger(0);
    private final AtomicInteger checkpointsAnchored = new AtomicInteger(0);

    TrainingRunRecord(String runId, String submitterId, int epochCount, byte[] configHash) {
        this.runId = Objects.requireNonNull(runId);
        this.submitterId = Objects.requireNonNull(submitterId);
        if (epochCount <= 0 || epochCount > 50000) throw new TP08ConfigValidationException("epochCount");
        this.epochCount = epochCount;
        this.configHash = configHash != null ? configHash.clone() : new byte[32];
        this.registeredAtEpochMillis = System.currentTimeMillis();
    }

    String getRunId() { return runId; }
    String getSubmitterId() { return submitterId; }
    int getEpochCount() { return epochCount; }
    byte[] getConfigHash() { return configHash.clone(); }
    long getRegisteredAtEpochMillis() { return registeredAtEpochMillis; }
    boolean isArchived() { return archived; }
    void setArchived(boolean v) { this.archived = v; }
    int getEpochsRecorded() { return epochsRecorded.get(); }
    int incrementEpochsRecorded() { return epochsRecorded.incrementAndGet(); }
    int getCheckpointsAnchored() { return checkpointsAnchored.get(); }
    int incrementCheckpointsAnchored() { return checkpointsAnchored.incrementAndGet(); }
}

// -----------------------------------------------------------------------------
// EPOCH RECORD
// -----------------------------------------------------------------------------

final class EpochRecord {
    private final String runId;
    private final int epochIndex;
    private final long lossScaled;
    private final byte[] gradientRoot;
    private final long recordedAtEpochMillis;

    EpochRecord(String runId, int epochIndex, long lossScaled, byte[] gradientRoot) {
        this.runId = runId;
        this.epochIndex = epochIndex;
        this.lossScaled = lossScaled;
        this.gradientRoot = gradientRoot != null ? gradientRoot.clone() : new byte[32];
        this.recordedAtEpochMillis = System.currentTimeMillis();
    }

    String getRunId() { return runId; }
    int getEpochIndex() { return epochIndex; }
    long getLossScaled() { return lossScaled; }
    double getLoss() { return (double) lossScaled / TP08Constants.LOSS_SCALE_FACTOR; }
    byte[] getGradientRoot() { return gradientRoot.clone(); }
    long getRecordedAtEpochMillis() { return recordedAtEpochMillis; }
}

// -----------------------------------------------------------------------------
// CHECKPOINT RECORD
// -----------------------------------------------------------------------------

final class CheckpointRecord {
    private final String runId;
    private final int checkpointIndex;
    private final byte[] stateHash;
    private final long anchoredAtEpochMillis;

    CheckpointRecord(String runId, int checkpointIndex, byte[] stateHash) {
        this.runId = runId;
        this.checkpointIndex = checkpointIndex;
        this.stateHash = stateHash != null ? stateHash.clone() : new byte[32];
        this.anchoredAtEpochMillis = System.currentTimeMillis();
    }

    String getRunId() { return runId; }
    int getCheckpointIndex() { return checkpointIndex; }
    byte[] getStateHash() { return stateHash.clone(); }
    long getAnchoredAtEpochMillis() { return anchoredAtEpochMillis; }
}

// -----------------------------------------------------------------------------
// TRAINING CONFIG
// -----------------------------------------------------------------------------

final class TrainingConfig {
    private final int maxEpochs;
    private final int batchSize;
    private final double learningRate;
    private final double gradientClipNorm;
    private final int checkpointEveryEpochs;
    private final long randomSeed;
    private final String optimizerName;
    private final String lossName;

    TrainingConfig(int maxEpochs, int batchSize, double learningRate,
                   double gradientClipNorm, int checkpointEveryEpochs,
                   long randomSeed, String optimizerName, String lossName) {
        this.maxEpochs = maxEpochs;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.gradientClipNorm = gradientClipNorm;
        this.checkpointEveryEpochs = checkpointEveryEpochs;
        this.randomSeed = randomSeed;
        this.optimizerName = optimizerName != null ? optimizerName : "Adam";
        this.lossName = lossName != null ? lossName : "MSE";
    }

    int getMaxEpochs() { return maxEpochs; }
    int getBatchSize() { return batchSize; }
    double getLearningRate() { return learningRate; }
    double getGradientClipNorm() { return gradientClipNorm; }
    int getCheckpointEveryEpochs() { return checkpointEveryEpochs; }
    long getRandomSeed() { return randomSeed; }
    String getOptimizerName() { return optimizerName; }
    String getLossName() { return lossName; }

    static Builder builder() { return new Builder(); }
    static final class Builder {
        private int maxEpochs = TP08Constants.MAX_EPOCHS_DEFAULT;
        private int batchSize = TP08Constants.BATCH_SIZE_DEFAULT;
        private double learningRate = TP08Constants.LEARNING_RATE_DEFAULT;
        private double gradientClipNorm = TP08Constants.GRADIENT_CLIP_NORM;
        private int checkpointEveryEpochs = TP08Constants.CHECKPOINT_EVERY_EPOCHS;
        private long randomSeed = TP08Constants.RANDOM_SEED_BASE + System.nanoTime();
        private String optimizerName = "Adam";
        private String lossName = "MSE";
        Builder maxEpochs(int v) { this.maxEpochs = v; return this; }
        Builder batchSize(int v) { this.batchSize = v; return this; }
        Builder learningRate(double v) { this.learningRate = v; return this; }
        Builder gradientClipNorm(double v) { this.gradientClipNorm = v; return this; }
        Builder checkpointEveryEpochs(int v) { this.checkpointEveryEpochs = v; return this; }
        Builder randomSeed(long v) { this.randomSeed = v; return this; }
        Builder optimizerName(String v) { this.optimizerName = v; return this; }
        Builder lossName(String v) { this.lossName = v; return this; }
        TrainingConfig build() {
            return new TrainingConfig(maxEpochs, batchSize, learningRate,
                    gradientClipNorm, checkpointEveryEpochs, randomSeed, optimizerName, lossName);
        }
    }
}

// -----------------------------------------------------------------------------
// LOSS INTERFACE & IMPLEMENTATIONS
// -----------------------------------------------------------------------------

interface LossFunction {
    double compute(double[] predicted, double[] target);
    void computeGradient(double[] predicted, double[] target, double[] gradientOut);
    String name();
}

final class MSELoss implements LossFunction {
    @Override public double compute(double[] pred, double[] target) {
        if (pred.length != target.length) throw new IllegalArgumentException("length mismatch");
        double sum = 0;
        for (int i = 0; i < pred.length; i++) {
            double d = pred[i] - target[i];
            sum += d * d;
        }
        return sum / pred.length;
    }
    @Override public void computeGradient(double[] pred, double[] target, double[] gradientOut) {
        int n = pred.length;
        for (int i = 0; i < n; i++)
            gradientOut[i] = 2.0 * (pred[i] - target[i]) / n;
    }
    @Override public String name() { return "MSE"; }
}

final class CrossEntropyLoss implements LossFunction {
    @Override public double compute(double[] pred, double[] target) {
        double sum = 0;
        for (int i = 0; i < pred.length; i++) {
            double p = Math.max(1e-15, Math.min(1 - 1e-15, pred[i]));
            sum -= target[i] * Math.log(p);
        }
        return sum / pred.length;
    }
    @Override public void computeGradient(double[] pred, double[] target, double[] gradientOut) {
        int n = pred.length;
        for (int i = 0; i < n; i++) {
            double p = Math.max(1e-15, Math.min(1 - 1e-15, pred[i]));
            gradientOut[i] = -(target[i] / p) / n;
        }
    }
    @Override public String name() { return "CrossEntropy"; }
}

final class HuberLoss implements LossFunction {
    private final double delta;
    HuberLoss(double delta) { this.delta = delta; }
    @Override public double compute(double[] pred, double[] target) {
        double sum = 0;
        for (int i = 0; i < pred.length; i++) {
            double d = pred[i] - target[i];
            double abs = Math.abs(d);
            sum += abs <= delta ? 0.5 * d * d : delta * (abs - 0.5 * delta);
        }
        return sum / pred.length;
    }
    @Override public void computeGradient(double[] pred, double[] target, double[] gradientOut) {
        int n = pred.length;
        for (int i = 0; i < n; i++) {
            double d = pred[i] - target[i];
            if (Math.abs(d) <= delta) gradientOut[i] = d / n;
            else gradientOut[i] = (delta * Math.signum(d)) / n;
        }
    }
    @Override public String name() { return "Huber"; }
}

// -----------------------------------------------------------------------------
// OPTIMIZER INTERFACE & IMPLEMENTATIONS
// -----------------------------------------------------------------------------

interface Optimizer {
    void step(double[] params, double[] gradients, int stepIndex);
    String name();
}

final class SGDOptimizer implements Optimizer {
    private final double lr;
    private final double momentum;
    private final double[] velocity;
    SGDOptimizer(double lr, double momentum, int paramLen) {
        this.lr = lr;
        this.momentum = momentum;
        this.velocity = new double[paramLen];
    }
    @Override public void step(double[] params, double[] gradients, int stepIndex) {
        for (int i = 0; i < params.length; i++) {
            velocity[i] = momentum * velocity[i] + gradients[i];
            params[i] -= lr * velocity[i];
        }
    }
    @Override public String name() { return "SGD"; }
}

final class AdamOptimizer implements Optimizer {
    private final double lr;
    private final double beta1;
    private final double beta2;
    private final double eps;
    private final double[] m;
    private final double[] v;
    private int t = 0;
    AdamOptimizer(double lr, double beta1, double beta2, double eps, int paramLen) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.m = new double[paramLen];
        this.v = new double[paramLen];
    }
    @Override public void step(double[] params, double[] gradients, int stepIndex) {
        t++;
        for (int i = 0; i < params.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];
            double mHat = m[i] / (1 - Math.pow(beta1, t));
            double vHat = v[i] / (1 - Math.pow(beta2, t));
            params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }
    @Override public String name() { return "Adam"; }
}

final class RMSpropOptimizer implements Optimizer {
    private final double lr;
    private final double decay;
    private final double[] cache;
    RMSpropOptimizer(double lr, double decay, int paramLen) {
        this.lr = lr;
        this.decay = decay;
        this.cache = new double[paramLen];
    }
    @Override public void step(double[] params, double[] gradients, int stepIndex) {
        for (int i = 0; i < params.length; i++) {
            cache[i] = decay * cache[i] + (1 - decay) * gradients[i] * gradients[i];
            params[i] -= lr * gradients[i] / (Math.sqrt(cache[i]) + 1e-8);
        }
    }
    @Override public String name() { return "RMSprop"; }
}

// -----------------------------------------------------------------------------
// GRADIENT UTILS
// -----------------------------------------------------------------------------

final class GradientUtils {
    static double norm(double[] g) {
        double sum = 0;
        for (double v : g) sum += v * v;
        return Math.sqrt(sum);
    }
    static void clipInPlace(double[] g, double maxNorm) {
        double n = norm(g);
        if (n > maxNorm && n > 0) {
            double scale = maxNorm / n;
            for (int i = 0; i < g.length; i++) g[i] *= scale;
        }
    }
    static byte[] hashForRoot(double[] gradient) {
        ByteBuffer bb = ByteBuffer.allocate(Double.BYTES * gradient.length);
        bb.order(ByteOrder.BIG_ENDIAN);
        for (double v : gradient) bb.putDouble(v);
        return Arrays.copyOf(MessageDigestHash.sha256(bb.array()), 32);
    }
    private GradientUtils() {}
}

final class MessageDigestHash {
    static byte[] sha256(byte[] input) {
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            return md.digest(input);
        } catch (Exception e) { throw new RuntimeException(e); }
    }
    private MessageDigestHash() {}
}

// -----------------------------------------------------------------------------
// DATASET & BATCH
// -----------------------------------------------------------------------------

interface Dataset {
    int size();
    void getBatch(int startIdx, int count, double[][] featuresOut, double[][] targetsOut);
    int featureDim();
    int targetDim();
}

final class ArrayDataset implements Dataset {
    private final double[][] features;
    private final double[][] targets;
    private final Random rng;

    ArrayDataset(double[][] features, double[][] targets, long seed) {
        if (features.length != targets.length || features.length == 0)
            throw new TP08DatasetEmptyException();
        this.features = features;
        this.targets = targets;
        this.rng = new Random(seed);
    }

    @Override public int size() { return features.length; }
    @Override public int featureDim() { return features[0].length; }
    @Override public int targetDim() { return targets[0].length; }

    @Override public void getBatch(int startIdx, int count, double[][] featuresOut, double[][] targetsOut) {
        int n = Math.min(count, features.length - startIdx);
        for (int i = 0; i < n; i++) {
            System.arraycopy(features[startIdx + i], 0, featuresOut[i], 0, features[0].length);
            System.arraycopy(targets[startIdx + i], 0, targetsOut[i], 0, targets[0].length);
        }
    }

    int[] shuffledIndices() {
        int[] idx = new int[features.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        for (int i = idx.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        return idx;
    }
}

// -----------------------------------------------------------------------------
