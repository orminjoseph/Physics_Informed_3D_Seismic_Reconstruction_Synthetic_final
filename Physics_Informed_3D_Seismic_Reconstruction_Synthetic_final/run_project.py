"""
======================================================================
MASTER PROJECT PIPELINE
======================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings

MASTER PROJECT ORCHESTRATOR

Purpose
-------
This script is the top-level controller for the complete research
pipeline.

The master pipeline coordinates:

    1. Training / checkpoint verification
    2. Model evaluation
    3. Reconstruction gallery generation
    4. Uncertainty analysis
    5. Uncertainty evaluation
    6. Uncertainty statistics
    7. Common seven-method reconstruction comparison
    8. Controlled classical-baseline experiments
    9. Controlled proposed-model experiment
   10. Ablation study
   11. Statistical significance analysis
   12. Thesis tables and final report

IMPORTANT
---------
This file is an ORCHESTRATOR.

Detailed scientific computation belongs to the appropriate module.

This file should NOT duplicate:

    - model architecture
    - loss calculations
    - dataset generation
    - baseline algorithms
    - uncertainty calculations
    - metric calculations
    - ablation calculations
    - statistical tests
    - thesis-table calculations

The master pipeline only determines:

    what should run,
    in what order,
    and whether required outputs exist.

Training and evaluation are intentionally separated.

Training:
    dataset
        |
        v
    final model training
        |
        v
    best_model.pth

Evaluation:
    best_model.pth
        |
        v
    controlled evaluation matrix
        |
        v
    case-level results
        |
        v
    statistical analysis
        |
        v
    final report

The controlled evaluation matrix is NOT used to train 750 models.

Only ONE final model is trained.

The complete controlled matrix is defined by the evaluation
modules. This master script does not hard-code the matrix size.

Author: Ormin Joseph
======================================================================
"""

# ======================================================================
# STANDARD LIBRARY
# ======================================================================

from pathlib import Path
import sys
import traceback


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

from utils import config


# ======================================================================
# PROJECT ROOT
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ======================================================================
# CHECKPOINT PATH
# ======================================================================

BEST_CHECKPOINT = (
    Path(config.CHECKPOINT_DIR)
    / "best_model.pth"
)


# ======================================================================
# PIPELINE RESULTS
# ======================================================================

PIPELINE_RESULTS = {
    "configuration": False,
    "training": False,
    "checkpoint": False,
    "evaluation": False,
    "final_outputs": False,
}


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def print_header(title: str) -> None:
    """
    Print a clearly separated pipeline section header.
    """

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


def print_success(message: str) -> None:
    """
    Print a successful pipeline message.
    """

    print(f"[PASS] {message}")


def print_warning(message: str) -> None:
    """
    Print a warning without terminating the pipeline.
    """

    print(f"[WARNING] {message}")


def print_failure(message: str) -> None:
    """
    Print a pipeline failure message.
    """

    print(f"[FAIL] {message}")


# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================

def validate_configuration() -> bool:
    """
    Validate the minimum configuration required by the master pipeline.

    The detailed configuration validation remains the responsibility
    of utils.config.
    """

    print_header(
        "MASTER PIPELINE CONFIGURATION VALIDATION"
    )

    # ------------------------------------------------------------------
    # Experiment name
    # ------------------------------------------------------------------

    if not isinstance(
        config.EXPERIMENT_NAME,
        str,
    ) or not config.EXPERIMENT_NAME.strip():

        raise RuntimeError(
            "EXPERIMENT_NAME must be a non-empty string."
        )

    # ------------------------------------------------------------------
    # Dataset mode
    # ------------------------------------------------------------------

    if not isinstance(
        config.DATASET_MODE,
        str,
    ) or not config.DATASET_MODE.strip():

        raise RuntimeError(
            "DATASET_MODE must be a non-empty string."
        )

    # ------------------------------------------------------------------
    # Device policy
    # ------------------------------------------------------------------

    if not isinstance(
        config.DEVICE,
        str,
    ):

        raise RuntimeError(
            "DEVICE must be a string such as "
            "'cpu', 'cuda', or 'auto'."
        )

    device_policy = config.DEVICE.lower().strip()

    if device_policy not in {
        "auto",
        "cpu",
        "cuda",
    }:

        raise RuntimeError(
            "DEVICE must be one of: "
            "'auto', 'cpu', or 'cuda'."
        )

    # ------------------------------------------------------------------
    # Controlled-matrix case limit
    # ------------------------------------------------------------------
    #
    # None:
    #     run the complete controlled evaluation matrix.
    #
    # Integer:
    #     run only that number of cases for smoke testing.
    #
    # The actual number of factors/cases belongs to the evaluation
    # modules and is deliberately not hard-coded here.
    # ------------------------------------------------------------------

    case_limit = (
        config.CONTROLLED_MATRIX_CASE_LIMIT
    )

    if case_limit is not None:

        if not isinstance(
            case_limit,
            int,
        ):

            raise RuntimeError(
                "CONTROLLED_MATRIX_CASE_LIMIT must be "
                "an integer or None."
            )

        if case_limit <= 0:

            raise RuntimeError(
                "CONTROLLED_MATRIX_CASE_LIMIT must be "
                "greater than zero or None."
            )

    # ------------------------------------------------------------------
    # Create required top-level directories.
    # ------------------------------------------------------------------

    checkpoint_directory = Path(
        config.CHECKPOINT_DIR
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory = Path(
        config.REPORT_DIR
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Display configuration.
    # ------------------------------------------------------------------

    print(
        f"Experiment              : "
        f"{config.EXPERIMENT_NAME}"
    )

    print(
        f"Dataset mode             : "
        f"{config.DATASET_MODE}"
    )

    print(
        f"Configured device        : "
        f"{config.DEVICE}"
    )

    print(
        f"Run training             : "
        f"{config.RUN_TRAINING}"
    )

    print(
        f"Resume evaluation        : "
        f"{config.RESUME_EVALUATION}"
    )

    print(
        f"Force rerun evaluation   : "
        f"{config.FORCE_RERUN_EVALUATION}"
    )

    print(
        "Controlled matrix limit  : "
        f"{case_limit}"
    )

    print(
        f"Best checkpoint          : "
        f"{BEST_CHECKPOINT}"
    )

    print_success(
        "Master pipeline configuration is valid."
    )

    PIPELINE_RESULTS[
        "configuration"
    ] = True

    return True


# ======================================================================
# CHECKPOINT VERIFICATION
# ======================================================================

def verify_checkpoint() -> bool:
    """
    Verify that the final best-model checkpoint exists.

    Evaluation always uses:

        best_model.pth

    rather than the latest intermediate checkpoint.
    """

    print_header(
        "CHECKPOINT VERIFICATION"
    )

    checkpoint = Path(
        BEST_CHECKPOINT
    )

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    if not checkpoint.exists():

        raise FileNotFoundError(
            "Required best model checkpoint was not found:\n"
            f"{checkpoint}\n\n"
            "Train the final model first by setting "
            "RUN_TRAINING=True."
        )

    # ------------------------------------------------------------------
    # File-size check
    # ------------------------------------------------------------------

    if checkpoint.stat().st_size <= 0:

        raise RuntimeError(
            "The best model checkpoint exists but is empty:\n"
            f"{checkpoint}"
        )

    # ------------------------------------------------------------------
    # Display checkpoint information
    # ------------------------------------------------------------------

    print(
        "Checkpoint found:"
    )

    print(
        f"    {checkpoint}"
    )

    print(
        "Checkpoint size:"
    )

    print(
        f"    {checkpoint.stat().st_size:,} bytes"
    )

    print_success(
        "Best model checkpoint verified."
    )

    PIPELINE_RESULTS[
        "checkpoint"
    ] = True

    return True


# ======================================================================
# TRAINING STAGE
# ======================================================================

def run_training_stage() -> bool:
    """
    Execute the final model training stage.

    The training module is imported lazily.

    This is intentional because when:

        RUN_TRAINING=False

    the master pipeline must be able to evaluate an already-trained
    model without requiring the training entry point to be imported
    during startup.

    The actual training entry point is:

        train_final_model()

    not:

        main()
    """

    print_header(
        "TRAINING STAGE"
    )

    # ------------------------------------------------------------------
    # Training disabled
    # ------------------------------------------------------------------

    if not config.RUN_TRAINING:

        print(
            "RUN_TRAINING=False"
        )

        print(
            "Training stage skipped."
        )

        print(
            "The existing best_model.pth will be used."
        )

        PIPELINE_RESULTS[
            "training"
        ] = True

        return True

    # ------------------------------------------------------------------
    # Lazy import of the actual training entry point.
    #
    # The user's actual train/train_model.py defines:
    #
    #     train_final_model()
    #
    # It does not define main().
    # ------------------------------------------------------------------

    from train.train_model import (
        train_final_model
    )

    # ------------------------------------------------------------------
    # Execute final-model training.
    # ------------------------------------------------------------------

    train_final_model()

    print_success(
        "Final model training completed."
    )

    PIPELINE_RESULTS[
        "training"
    ] = True

    return True


# ======================================================================
# EVALUATION STAGE
# ======================================================================

def run_evaluation_stage() -> bool:
    """
    Execute the complete downstream evaluation pipeline.

    Detailed scientific evaluation remains inside:

        evaluation.run_full_evaluation

    The master pipeline deliberately does not duplicate the individual
    evaluation stages.
    """

    print_header(
        "COMPLETE EVALUATION STAGE"
    )

    # ------------------------------------------------------------------
    # The evaluation stage requires a valid trained checkpoint.
    # ------------------------------------------------------------------

    verify_checkpoint()

    # ------------------------------------------------------------------
    # Import the evaluation controller only when evaluation begins.
    # ------------------------------------------------------------------

    from evaluation.run_full_evaluation import (
        run_full_evaluation
    )

    # ------------------------------------------------------------------
    # Execute the complete evaluation sequence.
    # ------------------------------------------------------------------

    run_full_evaluation()

    print_success(
        "Complete evaluation pipeline finished."
    )

    PIPELINE_RESULTS[
        "evaluation"
    ] = True

    return True


# ======================================================================
# FINAL OUTPUT VERIFICATION
# ======================================================================

def verify_final_outputs() -> bool:
    """
    Perform final master-level output verification.

    Individual scientific evaluation modules are responsible for
    validating their own detailed output files.

    Therefore this function verifies only the critical project-level
    artifacts:

        1. best_model.pth exists and is non-empty.
        2. evaluation completed successfully.
        3. report directory exists.

    This function deliberately does NOT require a specific filename
    such as final_report.txt because the detailed evaluation controller
    owns the report-generation implementation.
    """

    print_header(
        "FINAL OUTPUT VERIFICATION"
    )

    # ------------------------------------------------------------------
    # Verify best checkpoint.
    # ------------------------------------------------------------------

    checkpoint = Path(
        BEST_CHECKPOINT
    )

    if not checkpoint.exists():

        raise RuntimeError(
            "Final best-model checkpoint is missing:\n"
            f"{checkpoint}"
        )

    if checkpoint.stat().st_size <= 0:

        raise RuntimeError(
            "Final best-model checkpoint is empty:\n"
            f"{checkpoint}"
        )

    # ------------------------------------------------------------------
    # Verify evaluation stage.
    # ------------------------------------------------------------------

    if not PIPELINE_RESULTS[
        "evaluation"
    ]:

        raise RuntimeError(
            "Evaluation stage did not complete successfully."
        )

    # ------------------------------------------------------------------
    # Verify report directory.
    # ------------------------------------------------------------------

    report_directory = Path(
        config.REPORT_DIR
    )

    if not report_directory.exists():

        raise RuntimeError(
            "Report directory does not exist:\n"
            f"{report_directory}"
        )

    # ------------------------------------------------------------------
    # Display successful verification.
    # ------------------------------------------------------------------

    print(
        "Best checkpoint verified."
    )

    print(
        "Evaluation stage verified."
    )

    print(
        "Report directory verified:"
    )

    print(
        f"    {report_directory}"
    )

    print_success(
        "Master-level final output verification passed."
    )

    PIPELINE_RESULTS[
        "final_outputs"
    ] = True

    return True


# ======================================================================
# PIPELINE SUMMARY
# ======================================================================

def print_pipeline_summary() -> None:
    """
    Print the final status of each master-pipeline stage.
    """

    print_header(
        "MASTER PIPELINE SUMMARY"
    )

    status_names = {
        "configuration": "Configuration",
        "training": "Training",
        "checkpoint": "Checkpoint",
        "evaluation": "Evaluation",
        "final_outputs": "Final outputs",
    }

    for key, label in status_names.items():

        status = PIPELINE_RESULTS.get(
            key,
            False,
        )

        if status:

            print(
                f"[PASS] {label}"
            )

        else:

            print(
                f"[FAIL] {label}"
            )

    print()

    print(
        f"Experiment : "
        f"{config.EXPERIMENT_NAME}"
    )

    print(
        f"Dataset    : "
        f"{config.DATASET_MODE}"
    )

    print(
        f"Checkpoint : "
        f"{BEST_CHECKPOINT}"
    )

    print()


# ======================================================================
# MASTER PROJECT RUNNER
# ======================================================================

def run_project() -> bool:
    """
    Execute the complete research project pipeline.

    Sequence
    --------
    1. Validate configuration.
    2. Train final model if requested.
    3. Verify best checkpoint.
    4. Execute complete evaluation.
    5. Verify project-level outputs.
    6. Print final summary.

    Returns
    -------
    bool
        True when the complete pipeline succeeds.
        False when an exception occurs.
    """

    print()
    print("=" * 78)
    print(
        "PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION"
    )
    print(
        "MASTER PROJECT PIPELINE"
    )
    print("=" * 78)
    print()

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Experiment   : "
        f"{config.EXPERIMENT_NAME}"
    )

    print(
        f"Dataset mode : "
        f"{config.DATASET_MODE}"
    )

    print()

    try:

        # ==============================================================
        # STAGE 1
        # CONFIGURATION
        # ==============================================================

        validate_configuration()

        # ==============================================================
        # STAGE 2
        # TRAINING
        # ==============================================================

        run_training_stage()

        # ==============================================================
        # STAGE 3
        # CHECKPOINT
        # ==============================================================

        verify_checkpoint()

        # ==============================================================
        # STAGE 4
        # COMPLETE EVALUATION
        # ==============================================================

        run_evaluation_stage()

        # ==============================================================
        # STAGE 5
        # FINAL OUTPUT VERIFICATION
        # ==============================================================

        verify_final_outputs()

        # ==============================================================
        # FINAL SUMMARY
        # ==============================================================

        print_pipeline_summary()

        print()
        print("=" * 78)
        print(
            "MASTER PROJECT PIPELINE COMPLETED SUCCESSFULLY"
        )
        print("=" * 78)
        print()

        return True

    except Exception as error:

        print()
        print("=" * 78)
        print(
            "MASTER PROJECT PIPELINE FAILED"
        )
        print("=" * 78)
        print()

        print_failure(
            str(error)
        )

        print()
        print(
            "Traceback:"
        )

        traceback.print_exc()

        print()

        print_pipeline_summary()

        return False


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    success = run_project()

    # ------------------------------------------------------------------
    # Return a non-zero exit code when the pipeline fails.
    # ------------------------------------------------------------------

    if not success:

        sys.exit(1)

    sys.exit(0)