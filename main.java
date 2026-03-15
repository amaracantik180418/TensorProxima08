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
