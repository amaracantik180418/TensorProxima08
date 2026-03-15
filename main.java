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
// MODEL INTERFACE (simple linear for demo)
// -----------------------------------------------------------------------------

interface Model {
    void forward(double[][] input, double[][] output);
    void backward(double[][] input, double[][] outputGrad, double[][] paramGrad);
    double[] getParams();
    void setParams(double[] params);
    int paramCount();
}

final class LinearModel implements Model {
    private final int inDim;
    private final int outDim;
    private final double[] params; // [outDim * (inDim + 1)]: row-major weight then bias
    private double[][] lastInput;

    LinearModel(int inDim, int outDim, Random rng) {
        this.inDim = inDim;
        this.outDim = outDim;
        this.params = new double[outDim * (inDim + 1)];
        double scale = 1.0 / Math.sqrt(inDim + 1);
        for (int i = 0; i < params.length; i++)
            params[i] = (rng.nextDouble() * 2 - 1) * scale;
    }

    @Override public int paramCount() { return params.length; }
    @Override public double[] getParams() { return params.clone(); }
    @Override public void setParams(double[] p) { System.arraycopy(p, 0, params, 0, Math.min(p.length, params.length)); }

    @Override public void forward(double[][] input, double[][] output) {
        lastInput = input;
        int batch = input.length;
        for (int b = 0; b < batch; b++) {
            for (int o = 0; o < outDim; o++) {
                double sum = params[outDim * (inDim + 1) - outDim + o];
                for (int i = 0; i < inDim; i++)
                    sum += input[b][i] * params[o * (inDim + 1) + i];
                output[b][o] = sum;
            }
        }
    }

    @Override public void backward(double[][] input, double[][] outputGrad, double[][] paramGrad) {
        int batch = input.length;
        Arrays.stream(paramGrad).forEach(r -> Arrays.fill(r, 0));
        for (int b = 0; b < batch; b++) {
            for (int o = 0; o < outDim; o++) {
                double g = outputGrad[b][o];
                for (int i = 0; i < inDim; i++)
                    paramGrad[o][i] += g * input[b][i];
                paramGrad[o][inDim] += g;
            }
        }
        for (int o = 0; o < outDim; o++)
            for (int i = 0; i <= inDim; i++)
                paramGrad[o][i] /= batch;
    }
}

// -----------------------------------------------------------------------------
// METRICS
// -----------------------------------------------------------------------------

final class EpochMetrics {
    private final int epochIndex;
    private final double loss;
    private final long durationMs;
    private final int batchesProcessed;

    EpochMetrics(int epochIndex, double loss, long durationMs, int batchesProcessed) {
        this.epochIndex = epochIndex;
        this.loss = loss;
        this.durationMs = durationMs;
        this.batchesProcessed = batchesProcessed;
    }
    int getEpochIndex() { return epochIndex; }
    double getLoss() { return loss; }
    long getDurationMs() { return durationMs; }
    int getBatchesProcessed() { return batchesProcessed; }
    @Override public String toString() {
        return String.format("EpochMetrics{epoch=%d, loss=%.6f, durationMs=%d, batches=%d}",
                epochIndex, loss, durationMs, batchesProcessed);
    }
}

// -----------------------------------------------------------------------------
// RUN REGISTRY (in-memory)
// -----------------------------------------------------------------------------

final class RunRegistry {
    private final Map<String, TrainingRunRecord> runs = new ConcurrentHashMap<>();
    private final Map<String, List<EpochRecord>> epochsByRun = new ConcurrentHashMap<>();
    private final Map<String, List<CheckpointRecord>> checkpointsByRun = new ConcurrentHashMap<>();
    private final List<String> runIdOrder = new CopyOnWriteArrayList<>();

    String registerRun(String submitterId, int epochCount, byte[] configHash) {
        String runId = TP08Constants.RUN_ID_PREFIX + UUID.randomUUID().toString().replace("-", "").substring(0, 16);
        TrainingRunRecord r = new TrainingRunRecord(runId, submitterId, epochCount, configHash);
        runs.put(runId, r);
        runIdOrder.add(runId);
        epochsByRun.put(runId, new CopyOnWriteArrayList<>());
        checkpointsByRun.put(runId, new CopyOnWriteArrayList<>());
        return runId;
    }

    TrainingRunRecord getRun(String runId) {
        TrainingRunRecord r = runs.get(runId);
        if (r == null) throw new TP08RunNotFoundException(runId);
        return r;
    }

    void recordEpoch(String runId, int epochIndex, long lossScaled, byte[] gradientRoot) {
        TrainingRunRecord r = getRun(runId);
        if (r.isArchived()) throw new TP08CheckpointException("Run archived");
        if (epochIndex >= r.getEpochCount()) throw new TP08EpochIndexException(epochIndex, r.getEpochCount());
        if (r.getEpochsRecorded() != epochIndex) throw new TP08CheckpointException("Epoch order");
        EpochRecord rec = new EpochRecord(runId, epochIndex, lossScaled, gradientRoot);
        epochsByRun.get(runId).add(rec);
        r.incrementEpochsRecorded();
    }

    void anchorCheckpoint(String runId, int checkpointIndex, byte[] stateHash) {
        TrainingRunRecord r = getRun(runId);
        if (r.isArchived()) throw new TP08CheckpointException("Run archived");
        CheckpointRecord rec = new CheckpointRecord(runId, checkpointIndex, stateHash);
        checkpointsByRun.get(runId).add(rec);
        r.incrementCheckpointsAnchored();
    }

    void archiveRun(String runId) {
        getRun(runId).setArchived(true);
    }

    List<String> getAllRunIds() { return new ArrayList<>(runIdOrder); }
    int totalRuns() { return runs.size(); }
    List<EpochRecord> getEpochs(String runId) { return new ArrayList<>(epochsByRun.getOrDefault(runId, List.of())); }
    List<CheckpointRecord> getCheckpoints(String runId) { return new ArrayList<>(checkpointsByRun.getOrDefault(runId, List.of())); }
}

// -----------------------------------------------------------------------------
// TRAINER BOT
// -----------------------------------------------------------------------------

class TrainerBot {
    protected final RunRegistry registry;
    protected final TrainingConfig config;
    protected final LossFunction lossFn;
    protected final Optimizer optimizer;
    protected final Model model;
    protected final Dataset dataset;
    protected final double gradientClipNorm;

    TrainerBot(RunRegistry registry, TrainingConfig config, LossFunction lossFn,
              Optimizer optimizer, Model model, Dataset dataset) {
        this.registry = registry;
        this.config = config;
        this.lossFn = lossFn;
        this.optimizer = optimizer;
        this.model = model;
        this.dataset = dataset;
        this.gradientClipNorm = config.getGradientClipNorm();
    }

    String startRun(String submitterId) {
        byte[] configHash = hashConfig(config);
        return registry.registerRun(submitterId, config.getMaxEpochs(), configHash);
    }

    private byte[] hashConfig(TrainingConfig c) {
        String s = c.getMaxEpochs() + "|" + c.getBatchSize() + "|" + c.getLearningRate()
                + "|" + c.getOptimizerName() + "|" + c.getLossName();
        return MessageDigestHash.sha256(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    void runTraining(String runId) {
        TrainingRunRecord r = registry.getRun(runId);
        int batchSize = config.getBatchSize();
        int featureDim = dataset.featureDim();
        int targetDim = dataset.targetDim();
        int batchCount = (dataset.size() + batchSize - 1) / batchSize;
        double[][] batchFeatures = new double[batchSize][featureDim];
        double[][] batchTargets = new double[batchSize][targetDim];
        double[][] batchOutput = new double[batchSize][targetDim];
        double[][] outputGrad = new double[batchSize][targetDim];
        int globalStep = 0;
        for (int epoch = 0; epoch < config.getMaxEpochs(); epoch++) {
            long startMs = System.currentTimeMillis();
            double epochLoss = 0;
            int[] indices = dataset instanceof ArrayDataset
                    ? ((ArrayDataset) dataset).shuffledIndices()
                    : range(dataset.size());
            for (int b = 0; b < batchCount; b++) {
                int start = b * batchSize;
                int len = Math.min(batchSize, dataset.size() - start);
                if (len <= 0) continue;
                fillBatch(dataset, indices, start, len, batchFeatures, batchTargets);
                model.forward(batchFeatures, batchOutput);
                double batchLoss = 0;
                for (int i = 0; i < len; i++) {
                    batchLoss += lossFn.compute(batchOutput[i], batchTargets[i]);
                    lossFn.computeGradient(batchOutput[i], batchTargets[i], outputGrad[i]);
                }
                batchLoss /= len;
                epochLoss += batchLoss;
                double[] params = model.getParams();
                double[] grad = new double[params.length];
                model.backward(batchFeatures, outputGrad, reshapeGrad(grad, model));
                GradientUtils.clipInPlace(grad, gradientClipNorm);
                if (GradientUtils.norm(grad) > gradientClipNorm * 100) throw new TP08GradientExplosionException(GradientUtils.norm(grad));
                optimizer.step(params, grad, globalStep);
                model.setParams(params);
                globalStep++;
            }
            epochLoss /= batchCount;
            long lossScaled = (long) (epochLoss * TP08Constants.LOSS_SCALE_FACTOR);
            byte[] gradientRoot = GradientUtils.hashForRoot(model.getParams());
            registry.recordEpoch(runId, epoch, lossScaled, gradientRoot);
            if ((epoch + 1) % config.getCheckpointEveryEpochs() == 0) {
                byte[] paramBytes = new byte[model.paramCount() * Double.BYTES];
                ByteBuffer.wrap(paramBytes).order(ByteOrder.BIG_ENDIAN).asDoubleBuffer().put(model.getParams());
                byte[] stateHash = MessageDigestHash.sha256(paramBytes);
                registry.anchorCheckpoint(runId, (epoch + 1) / config.getCheckpointEveryEpochs() - 1, stateHash);
            }
            long durationMs = System.currentTimeMillis() - startMs;
            EpochMetrics m = new EpochMetrics(epoch, epochLoss, durationMs, batchCount);
            if (epoch % 10 == 0) System.out.println(m);
        }
    }

    private int[] range(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = i;
        return a;
    }

    private void fillBatch(Dataset ds, int[] indices, int start, int len, double[][] featOut, double[][] tgtOut) {
        for (int i = 0; i < len; i++) {
            int idx = indices[start + i];
            ds.getBatch(idx, 1, new double[][]{featOut[i]}, new double[][]{tgtOut[i]});
        }
    }

    private double[][] reshapeGrad(double[] flat, Model model) {
        int pc = model.paramCount();
        int cols = (int) Math.sqrt(pc);
        if (cols * cols != pc) cols = pc;
        int rows = (pc + cols - 1) / cols;
        double[][] out = new double[rows][cols];
        for (int i = 0; i < pc; i++) out[i / cols][i % cols] = flat[i];
        return out;
    }
}

// -----------------------------------------------------------------------------
// SCHEDULER BOT (learning rate schedule)
// -----------------------------------------------------------------------------

interface LRScheduler {
    double getLearningRate(int epoch, int step);
}

final class StepLRScheduler implements LRScheduler {
    private final double initialLr;
    private final int stepSize;
    private final double gamma;
    StepLRScheduler(double initialLr, int stepSize, double gamma) {
        this.initialLr = initialLr;
        this.stepSize = stepSize;
        this.gamma = gamma;
    }
    @Override public double getLearningRate(int epoch, int step) {
        int s = epoch * 1000 + step;
        return initialLr * Math.pow(gamma, s / stepSize);
    }
}

final class CosineAnnealingScheduler implements LRScheduler {
    private final double initialLr;
    private final int totalSteps;
    CosineAnnealingScheduler(double initialLr, int totalSteps) {
        this.initialLr = initialLr;
        this.totalSteps = totalSteps;
    }
    @Override public double getLearningRate(int epoch, int step) {
        int s = epoch * 1000 + step;
        if (s >= totalSteps) return initialLr * 0.01;
        return 0.5 * initialLr * (1 + Math.cos(Math.PI * s / totalSteps));
    }
}

// -----------------------------------------------------------------------------
// DATA AUGMENTATION (stub)
// -----------------------------------------------------------------------------

interface DataAugmentation {
    void apply(double[] feature);
}

final class NoOpAugmentation implements DataAugmentation {
    @Override public void apply(double[] feature) {}
}

final class GaussianNoiseAugmentation implements DataAugmentation {
    private final Random rng;
    private final double std;
    GaussianNoiseAugmentation(long seed, double std) {
        this.rng = new Random(seed);
        this.std = std;
    }
    @Override public void apply(double[] feature) {
        for (int i = 0; i < feature.length; i++)
            feature[i] += rng.nextGaussian() * std;
    }
}

// -----------------------------------------------------------------------------
// CHECKPOINT MANAGER
// -----------------------------------------------------------------------------

final class CheckpointManager {
    private final String baseDir;
    private final RunRegistry registry;

    CheckpointManager(String baseDir, RunRegistry registry) {
        this.baseDir = baseDir;
        this.registry = registry;
    }

    void saveCheckpoint(String runId, int checkpointIndex, Model model, TrainingConfig config) throws IOException {
        Path dir = Paths.get(baseDir, runId);
        Files.createDirectories(dir);
        Path file = dir.resolve("ckpt_" + checkpointIndex + ".bin");
        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(file))) {
            dos.writeInt(config.getMaxEpochs());
            dos.writeInt(config.getBatchSize());
            double[] p = model.getParams();
            dos.writeInt(p.length);
            for (double v : p) dos.writeDouble(v);
        }
    }

    void loadCheckpoint(String runId, int checkpointIndex, Model model) throws IOException {
        Path file = Paths.get(baseDir, runId, "ckpt_" + checkpointIndex + ".bin");
        if (!Files.exists(file)) throw new TP08CheckpointException("File not found: " + file);
        try (DataInputStream dis = new DataInputStream(Files.newInputStream(file))) {
            dis.readInt();
            dis.readInt();
            int n = dis.readInt();
            double[] p = new double[n];
            for (int i = 0; i < n; i++) p[i] = dis.readDouble();
            model.setParams(p);
        }
    }
}

// -----------------------------------------------------------------------------
// LOGGER
// -----------------------------------------------------------------------------

final class TP08Logger {
    private final String runId;
    private final List<String> lines = new CopyOnWriteArrayList<>();

    TP08Logger(String runId) { this.runId = runId; }
    void log(String level, String msg) {
        String line = Instant.now() + " [" + runId + "] [" + level + "] " + msg;
        lines.add(line);
        System.out.println(line);
    }
    void info(String msg) { log("INFO", msg); }
    void warn(String msg) { log("WARN", msg); }
    void error(String msg) { log("ERROR", msg); }
    List<String> getLines() { return new ArrayList<>(lines); }
    void writeToFile(Path path) throws IOException {
        Files.write(path, lines, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
}

// -----------------------------------------------------------------------------
// METRICS AGGREGATOR
// -----------------------------------------------------------------------------

final class MetricsAggregator {
    private final List<EpochMetrics> history = new CopyOnWriteArrayList<>();

    void add(EpochMetrics m) { history.add(m); }
    double getBestLoss() {
        return history.stream().mapToDouble(EpochMetrics::getLoss).min().orElse(Double.POSITIVE_INFINITY);
    }
    double getLastLoss() {
        return history.isEmpty() ? Double.NaN : history.get(history.size() - 1).getLoss();
    }
    List<EpochMetrics> getHistory() { return new ArrayList<>(history); }
}

// -----------------------------------------------------------------------------
// EARLY STOPPING
// -----------------------------------------------------------------------------

final class EarlyStoppingHandler {
    private final int patience;
    private final double minDelta;
    private int waitCount = 0;
    private double bestLoss = Double.POSITIVE_INFINITY;

    EarlyStoppingHandler(int patience, double minDelta) {
        this.patience = patience;
        this.minDelta = minDelta;
    }

    boolean shouldStop(double currentLoss) {
        if (currentLoss < bestLoss - minDelta) {
            bestLoss = currentLoss;
            waitCount = 0;
            return false;
        }
        waitCount++;
        return waitCount >= patience;
    }

    void reset() {
        waitCount = 0;
        bestLoss = Double.POSITIVE_INFINITY;
    }
}

// -----------------------------------------------------------------------------
// VALIDATION LOOP
// -----------------------------------------------------------------------------

final class ValidationEvaluator {
    private final Model model;
    private final Dataset validationSet;
    private final LossFunction lossFn;

    ValidationEvaluator(Model model, Dataset validationSet, LossFunction lossFn) {
        this.model = model;
        this.validationSet = validationSet;
        this.lossFn = lossFn;
    }

    double evaluate() {
        int n = validationSet.size();
        if (n == 0) return Double.NaN;
        double[][] feat = new double[1][validationSet.featureDim()];
        double[][] tgt = new double[1][validationSet.targetDim()];
        double[][] out = new double[1][validationSet.targetDim()];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            validationSet.getBatch(i, 1, feat, tgt);
            model.forward(feat, out);
            sum += lossFn.compute(out[0], tgt[0]);
        }
        return sum / n;
    }
}

// -----------------------------------------------------------------------------
// CALLBACK INTERFACE
// -----------------------------------------------------------------------------

interface TrainingCallback {
    void onEpochStart(String runId, int epoch);
    void onEpochEnd(String runId, int epoch, EpochMetrics metrics);
    void onCheckpoint(String runId, int checkpointIndex);
    void onRunComplete(String runId);
}

final class LoggingCallback implements TrainingCallback {
    private final TP08Logger logger;
    LoggingCallback(TP08Logger logger) { this.logger = logger; }
    @Override public void onEpochStart(String runId, int epoch) { logger.info("Epoch start: " + epoch); }
    @Override public void onEpochEnd(String runId, int epoch, EpochMetrics metrics) { logger.info("Epoch end: " + metrics); }
    @Override public void onCheckpoint(String runId, int checkpointIndex) { logger.info("Checkpoint: " + checkpointIndex); }
    @Override public void onRunComplete(String runId) { logger.info("Run complete: " + runId); }
}

final class CompositeCallback implements TrainingCallback {
    private final List<TrainingCallback> callbacks = new ArrayList<>();
    void add(TrainingCallback c) { callbacks.add(c); }
    @Override public void onEpochStart(String runId, int epoch) { for (TrainingCallback c : callbacks) c.onEpochStart(runId, epoch); }
    @Override public void onEpochEnd(String runId, int epoch, EpochMetrics metrics) { for (TrainingCallback c : callbacks) c.onEpochEnd(runId, epoch, metrics); }
    @Override public void onCheckpoint(String runId, int checkpointIndex) { for (TrainingCallback c : callbacks) c.onCheckpoint(runId, checkpointIndex); }
    @Override public void onRunComplete(String runId) { for (TrainingCallback c : callbacks) c.onRunComplete(runId); }
}

// -----------------------------------------------------------------------------
// CONFIG SERIALIZER
// -----------------------------------------------------------------------------

final class ConfigSerializer {
    static String toJson(TrainingConfig c) {
        return String.format(
            "{\"maxEpochs\":%d,\"batchSize\":%d,\"learningRate\":%.10f,\"gradientClipNorm\":%.4f," +
            "\"checkpointEveryEpochs\":%d,\"randomSeed\":%d,\"optimizerName\":\"%s\",\"lossName\":\"%s\"}",
            c.getMaxEpochs(), c.getBatchSize(), c.getLearningRate(), c.getGradientClipNorm(),
            c.getCheckpointEveryEpochs(), c.getRandomSeed(), c.getOptimizerName(), c.getLossName()
        );
    }

    static TrainingConfig fromJson(String json) {
        Map<String, String> m = parseSimpleJson(json);
        return TrainingConfig.builder()
                .maxEpochs(Integer.parseInt(m.getOrDefault("maxEpochs", String.valueOf(TP08Constants.MAX_EPOCHS_DEFAULT))))
                .batchSize(Integer.parseInt(m.getOrDefault("batchSize", String.valueOf(TP08Constants.BATCH_SIZE_DEFAULT))))
                .learningRate(Double.parseDouble(m.getOrDefault("learningRate", String.valueOf(TP08Constants.LEARNING_RATE_DEFAULT))))
                .gradientClipNorm(Double.parseDouble(m.getOrDefault("gradientClipNorm", String.valueOf(TP08Constants.GRADIENT_CLIP_NORM))))
                .checkpointEveryEpochs(Integer.parseInt(m.getOrDefault("checkpointEveryEpochs", String.valueOf(TP08Constants.CHECKPOINT_EVERY_EPOCHS))))
                .randomSeed(Long.parseLong(m.getOrDefault("randomSeed", String.valueOf(TP08Constants.RANDOM_SEED_BASE))))
                .optimizerName(m.getOrDefault("optimizerName", "Adam"))
                .lossName(m.getOrDefault("lossName", "MSE"))
                .build();
    }

    private static Map<String, String> parseSimpleJson(String json) {
        Map<String, String> out = new HashMap<>();
        Pattern p = Pattern.compile("\"(\\w+)\"\\s*:\\s*(\"[^\"]*\"|\\d+\\.?\\d*|true|false)");
        Matcher matcher = p.matcher(json);
        while (matcher.find()) {
            String val = matcher.group(2);
            if (val.startsWith("\"")) val = val.substring(1, val.length() - 1);
            out.put(matcher.group(1), val);
        }
        return out;
    }
}

// -----------------------------------------------------------------------------
// RUN COMPARATOR
// -----------------------------------------------------------------------------

final class RunComparator {
    private final RunRegistry registry;

    RunComparator(RunRegistry registry) { this.registry = registry; }

    String getBestRunByLoss(List<String> runIds) {
        if (runIds.isEmpty()) return null;
        String best = runIds.get(0);
        double bestLoss = Double.POSITIVE_INFINITY;
        for (String id : runIds) {
            List<EpochRecord> epochs = registry.getEpochs(id);
            if (epochs.isEmpty()) continue;
            double last = epochs.get(epochs.size() - 1).getLoss();
            if (last < bestLoss) {
                bestLoss = last;
                best = id;
            }
        }
        return best;
    }

    Map<String, Double> getFinalLossPerRun(List<String> runIds) {
        Map<String, Double> out = new HashMap<>();
        for (String id : runIds) {
            List<EpochRecord> epochs = registry.getEpochs(id);
            if (epochs.isEmpty()) out.put(id, Double.NaN);
            else out.put(id, epochs.get(epochs.size() - 1).getLoss());
        }
        return out;
    }
}

// -----------------------------------------------------------------------------
// SYNTHETIC DATASET GENERATOR
// -----------------------------------------------------------------------------

final class SyntheticDatasetGenerator {
    private final Random rng;

    SyntheticDatasetGenerator(long seed) { this.rng = new Random(seed); }

    ArrayDataset generateLinear(int numSamples, int featureDim, int targetDim) {
        double[][] w = new double[targetDim][featureDim + 1];
        for (int i = 0; i < targetDim; i++)
            for (int j = 0; j <= featureDim; j++)
                w[i][j] = rng.nextDouble() * 2 - 1;
        double[][] features = new double[numSamples][featureDim];
        double[][] targets = new double[numSamples][targetDim];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < featureDim; j++) features[i][j] = rng.nextDouble() * 2 - 1;
            for (int o = 0; o < targetDim; o++) {
                double sum = w[o][featureDim];
                for (int j = 0; j < featureDim; j++) sum += features[i][j] * w[o][j];
                targets[i][o] = sum + 0.1 * rng.nextGaussian();
            }
        }
        return new ArrayDataset(features, targets, rng.nextLong());
    }

    ArrayDataset generateRandom(int numSamples, int featureDim, int targetDim) {
        double[][] features = new double[numSamples][featureDim];
        double[][] targets = new double[numSamples][targetDim];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < featureDim; j++) features[i][j] = rng.nextDouble();
            for (int j = 0; j < targetDim; j++) targets[i][j] = rng.nextDouble();
        }
        return new ArrayDataset(features, targets, rng.nextLong());
    }
}

// -----------------------------------------------------------------------------
// LOSS FACTORY
// -----------------------------------------------------------------------------

final class LossFactory {
    static LossFunction create(String name, Object... args) {
        switch (name == null ? "MSE" : name) {
            case "MSE": return new MSELoss();
            case "CrossEntropy": return new CrossEntropyLoss();
            case "Huber": return new HuberLoss(args.length > 0 ? ((Number) args[0]).doubleValue() : 1.0);
            default: return new MSELoss();
        }
    }
}

// -----------------------------------------------------------------------------
// OPTIMIZER FACTORY
// -----------------------------------------------------------------------------

final class OptimizerFactory {
    static Optimizer create(String name, double lr, int paramLen, Object... args) {
        switch (name == null ? "Adam" : name) {
            case "SGD":
                double momentum = args.length > 0 ? ((Number) args[0]).doubleValue() : 0.9;
                return new SGDOptimizer(lr, momentum, paramLen);
            case "Adam":
                double b1 = args.length > 0 ? ((Number) args[0]).doubleValue() : 0.9;
                double b2 = args.length > 1 ? ((Number) args[1]).doubleValue() : 0.999;
                double eps = args.length > 2 ? ((Number) args[2]).doubleValue() : 1e-8;
                return new AdamOptimizer(lr, b1, b2, eps, paramLen);
            case "RMSprop":
                double decay = args.length > 0 ? ((Number) args[0]).doubleValue() : 0.99;
                return new RMSpropOptimizer(lr, decay, paramLen);
            default: return new AdamOptimizer(lr, 0.9, 0.999, 1e-8, paramLen);
        }
    }
}

// -----------------------------------------------------------------------------
// BATCH ITERATOR
// -----------------------------------------------------------------------------

final class BatchIterator implements Iterator<int[]> {
    private final int totalSamples;
    private final int batchSize;
    private final int[] indices;
    private int position = 0;

    BatchIterator(int totalSamples, int batchSize, int[] indices) {
        this.totalSamples = totalSamples;
        this.batchSize = batchSize;
        this.indices = indices != null ? indices : range(totalSamples);
    }

    private static int[] range(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = i;
        return a;
    }

    @Override public boolean hasNext() { return position < totalSamples; }
    @Override public int[] next() {
        if (!hasNext()) throw new NoSuchElementException();
        int from = position;
        int to = Math.min(position + batchSize, totalSamples);
        int len = to - from;
        int[] batch = new int[len];
        for (int i = 0; i < len; i++) batch[i] = indices[from + i];
        position = to;
        return batch;
    }
}

// -----------------------------------------------------------------------------
// EXPORT RUN METADATA
// -----------------------------------------------------------------------------

final class RunMetadataExporter {
    static void exportToCsv(RunRegistry registry, String runId, Path path) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("epochIndex,lossScaled,loss,recordedAtMs");
        for (EpochRecord e : registry.getEpochs(runId)) {
            lines.add(String.format("%d,%d,%.10f,%d", e.getEpochIndex(), e.getLossScaled(), e.getLoss(), e.getRecordedAtEpochMillis()));
        }
        Files.write(path, lines, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    static void exportCheckpointsToCsv(RunRegistry registry, String runId, Path path) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add("checkpointIndex,anchoredAtMs");
        for (CheckpointRecord c : registry.getCheckpoints(runId)) {
            lines.add(String.format("%d,%d", c.getCheckpointIndex(), c.getAnchoredAtEpochMillis()));
        }
        Files.write(path, lines, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
}

// -----------------------------------------------------------------------------
// WARMUP RUNNER
// -----------------------------------------------------------------------------

final class WarmupRunner {
    static void warmup(Model model, Dataset dataset, int batchSize, int warmupBatches) {
        int fd = dataset.featureDim();
        int td = dataset.targetDim();
        double[][] feat = new double[batchSize][fd];
        double[][] tgt = new double[batchSize][td];
        double[][] out = new double[batchSize][td];
        for (int b = 0; b < warmupBatches; b++) {
            int len = Math.min(batchSize, dataset.size() - b * batchSize);
            if (len <= 0) break;
            dataset.getBatch(b * batchSize, len, feat, tgt);
            model.forward(feat, out);
        }
    }
}

// -----------------------------------------------------------------------------
// PARAMETER INITIALIZERS
// -----------------------------------------------------------------------------

interface ParamInitializer {
    void init(double[] params, int inDim, int outDim, Random rng);
}

final class XavierInitializer implements ParamInitializer {
    @Override public void init(double[] params, int inDim, int outDim, Random rng) {
        double scale = Math.sqrt(2.0 / (inDim + outDim));
        for (int i = 0; i < params.length; i++)
            params[i] = (rng.nextDouble() * 2 - 1) * scale;
    }
}

final class HeInitializer implements ParamInitializer {
    @Override public void init(double[] params, int inDim, int outDim, Random rng) {
        double scale = Math.sqrt(2.0 / inDim);
        for (int i = 0; i < params.length; i++)
            params[i] = rng.nextGaussian() * scale;
    }
}

final class ZeroInitializer implements ParamInitializer {
    @Override public void init(double[] params, int inDim, int outDim, Random rng) {
        Arrays.fill(params, 0);
    }
}

// -----------------------------------------------------------------------------
// LEARNING RATE FINDER (stub)
// -----------------------------------------------------------------------------

final class LRFinder {
    private final Model model;
    private final Dataset dataset;
    private final LossFunction lossFn;
    private final int batchSize;

    LRFinder(Model model, Dataset dataset, LossFunction lossFn, int batchSize) {
        this.model = model;
        this.dataset = dataset;
        this.lossFn = lossFn;
        this.batchSize = batchSize;
    }

    double suggestLr(int numSteps, double startLr, double endLr) {
        double lr = startLr;
        double bestLr = startLr;
        double bestLoss = Double.POSITIVE_INFINITY;
        double mult = Math.pow(endLr / startLr, 1.0 / numSteps);
        int fd = dataset.featureDim();
        int td = dataset.targetDim();
        double[][] feat = new double[batchSize][fd];
        double[][] tgt = new double[batchSize][td];
        double[][] out = new double[batchSize][td];
        for (int step = 0; step < numSteps; step++) {
            int start = (step * batchSize) % Math.max(1, dataset.size() - batchSize);
            dataset.getBatch(start, batchSize, feat, tgt);
            model.forward(feat, out);
            double loss = 0;
            for (int i = 0; i < batchSize; i++) loss += lossFn.compute(out[i], tgt[i]);
            loss /= batchSize;
