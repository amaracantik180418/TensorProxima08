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
