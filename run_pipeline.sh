#!/usr/bin/env bash

# Exit immediately if any command fails
set -euo pipefail

# ==============================================================================
# EMAIL NOTIFICATION SETTINGS (Set up your SMTP server details here)
# ==============================================================================
EMAIL_TO="saishanghlang20122002@gmail.com"
EMAIL_FROM="6631503129@lamduan.mfu.ac.th"
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=465
SMTP_USER="6631503129@lamduan.mfu.ac.th"
SMTP_PASS="dlfo psst iyol jicj"
# ==============================================================================

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

# Helper function to send email notification (cross-platform using Python)
send_error_email() {
    local stage_name="$1"
    local failed_step="$2"
    local log_file="$3"

    # Convert Windows backslashes to forward slashes to avoid Python escape issues
    log_file=$(echo "$log_file" | tr '\\' '/')

    # Get last 50 lines of log output to include in the email body
    local log_summary
    if [[ -f "$log_file" ]]; then
        log_summary=$(tail -n 50 "$log_file")
    else
        log_summary="No log file found."
    fi

    # Skip sending if email settings are not configured
    # Find available Python command (prioritize python and py for Windows compatibility)
    local python_cmd=""
    if command -v python &>/dev/null && python --version &>/dev/null; then
        python_cmd="python"
    elif command -v py &>/dev/null; then
        python_cmd="py"
    elif command -v python3 &>/dev/null; then
        python_cmd="python3"
    else
        echo "❌ Error: Python is not installed. Unable to send email notification."
        return 1
    fi

    if [[ -z "$EMAIL_TO" || -z "$SMTP_SERVER" || -z "$SMTP_USER" || -z "$SMTP_PASS" ]]; then
        echo "⚠️ Email notification skipped: SMTP parameters are not fully configured in run_pipeline.sh."
        return 0
    fi

    echo "📧 Sending error notification email..."

    # Create a temporary Python script to avoid heredoc issues with special characters
    local temp_py=$(mktemp /tmp/lulc_email.XXXXXX.py 2>/dev/null || mktemp lulc_email.XXXXXX.py)
    
    cat > "$temp_py" <<'EOF'
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

stage_name = sys.argv[1]
failed_step = sys.argv[2]
log_file = sys.argv[3]
email_from = sys.argv[4]
email_to = sys.argv[5]
smtp_server = sys.argv[6]
smtp_port = int(sys.argv[7])
smtp_user = sys.argv[8]
smtp_pass = sys.argv[9]

# Read log summary from stdin
log_summary = sys.stdin.read()

try:
    # Create email message
    msg = MIMEMultipart()
    msg['From'] = email_from
    msg['To'] = email_to
    msg['Subject'] = f"❌ [LULC Pipeline Failure] {stage_name}: {failed_step} Failed"

    body = f"""
Attention,

An error occurred in the LULC Segmentation training pipeline.

Failure Details:
---------------------------------------------
Stage: {stage_name}
Step: {failed_step}
Log Path: {log_file}
---------------------------------------------

Last 50 lines of log file:
---------------------------------------------
{log_summary}
---------------------------------------------
"""
    msg.attach(MIMEText(body, 'plain'))

    # Connect and send via SMTP (supports SSL on 465, STARTTLS on 587)
    if smtp_port == 465:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
    else:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls()
        server.ehlo()
    server.login(smtp_user, smtp_pass)
    server.sendmail(email_from, email_to, msg.as_string())
    server.close()
    print("✅ Email notification sent successfully.")
except Exception as e:
    print(f"❌ Failed to send email notification: {e}")
EOF

    # Send the email using the temporary script
    echo "$log_summary" | $python_cmd "$temp_py" "$stage_name" "$failed_step" "$log_file" "$EMAIL_FROM" "$EMAIL_TO" "$SMTP_SERVER" "$SMTP_PORT" "$SMTP_USER" "$SMTP_PASS"

    # Clean up the temporary script
    rm -f "$temp_py"
}



# Helper function to run training, evaluation, and visualization for a stage
run_stage() {
    local pipeline_name="$1"
    local stage_name="$2"
    local config_file="$3"
    local checkpoint_name="$4"
    
    local train_log="$LOG_DIR/${pipeline_name}_${stage_name}_train.log"
    local eval_log="$LOG_DIR/${pipeline_name}_${stage_name}_eval.log"
    local predict_log="$LOG_DIR/${pipeline_name}_${stage_name}_predict.log"
    local eval_output="outputs/evaluations/${pipeline_name}/${stage_name}/evaluation_results.json"
    local predict_output="outputs/predictions/${pipeline_name}/${stage_name}"

    echo "=========================================================================="
    echo "▶️ STARTING PIPELINE STAGE: $stage_name"
    echo "=========================================================================="
    
    # 1. Train the model
    echo "--- [1/3] Training ($stage_name) ---"
    echo "Running: python train.py --config $config_file"
    echo "Logging to: $train_log"
    if ! python train.py --config "$config_file" 2>&1 | tee "$train_log"; then
        echo "❌ ERROR: Training failed during stage '$stage_name'."
        echo "🔍 Please check the log file for details: $train_log"
        send_error_email "$stage_name" "Training" "$train_log"
        exit 1
    fi
    echo "Training completed successfully."
    echo


    # 2. Evaluate on test set
    echo "--- [2/3] Evaluating ($stage_name) ---"
    echo "Running: python evaluate.py --config $config_file --checkpoint checkpoints/$checkpoint_name --output $eval_output"
    echo "Logging to: $eval_log"
    if ! python evaluate.py --config "$config_file" --checkpoint "checkpoints/$checkpoint_name" --output "$eval_output" 2>&1 | tee "$eval_log"; then
        echo "❌ ERROR: Evaluation failed during stage '$stage_name'."
        echo "🔍 Please check the log file for details: $eval_log"
        send_error_email "$stage_name" "Evaluation" "$eval_log"
        exit 1
    fi
    echo "Evaluation completed successfully."
    echo


    # 3. Generate predictions & visualizations
    echo "--- [3/3] Predicting/Visualizing ($stage_name) ---"
    echo "Running: python predict.py --config $config_file --checkpoint checkpoints/$checkpoint_name --split test --output $predict_output"
    echo "Logging to: $predict_log"
    if ! python predict.py --config "$config_file" --checkpoint "checkpoints/$checkpoint_name" --split test --output "$predict_output" 2>&1 | tee "$predict_log"; then
        echo "❌ ERROR: Prediction/Visualization failed during stage '$stage_name'."
        echo "🔍 Please check the log file for details: $predict_log"
        send_error_email "$stage_name" "Prediction/Visualization" "$predict_log"
        exit 1
    fi
    echo "Predictions completed successfully."
    echo
    
    echo "✅ Pipeline Stage '$stage_name' finished successfully!"
    echo "=========================================================================="
    echo
}

# ==============================================================================
# PIPELINE EXECUTION FLOW
# ==============================================================================
PIPELINE_NAME="unetformer_resnet18"

Stage 1 — Baseline (IRSAMap only)
run_stage "$PIPELINE_NAME" "Stage_1_Baseline" \
          "configs/irsamap_unetformer_resnet18_8class.yaml" \
          "best_irsamap.pth"

# Stage 2 — Support training (IRSAMap + LoveDA support)
run_stage "$PIPELINE_NAME" "Stage_2_Support" \
          "configs/irsamap_loveda_support_unetformer_resnet18_8class.yaml" \
          "best_irsamap_loveda_support.pth"

# Stage 3 — Fine-tuning IRSAMap only from Stage 2 checkpoint
run_stage "$PIPELINE_NAME" "Stage_3_Finetuning" \
          "configs/irsamap_finetune_from_support.yaml" \
          "best_irsamap_finetune.pth"

echo "🎉 All 3 pipeline stages (Train → Evaluate → Predict) finished successfully!"
