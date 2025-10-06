# 🌞 Solar AI Pipeline - GitHub Actions Setup

This repository includes an automated CI/CD pipeline for continuous solar data collection, model training, and deployment.

## 🚀 Pipeline Overview

The GitHub Actions workflow automatically:

1. **📊 Data Collection** - Collects fresh solar images and storm event data daily
2. **🤖 Model Training** - Trains improved YOLO models when sufficient new data is available  
3. **🧪 Validation** - Tests model performance and validates accuracy thresholds
4. **🚀 Deployment** - Deploys high-performing models to production automatically
5. **📢 Notification** - Provides detailed reports and releases

## ⚙️ Setup Instructions

### 1. Repository Secrets

Add these secrets to your GitHub repository (`Settings` → `Secrets and variables` → `Actions`):

```bash
# Required for Google Cloud access
GCP_SA_KEY          # Google Cloud Service Account JSON key
```

### 2. Google Cloud Setup

1. **Create a Service Account** in your GCP project:
   ```bash
   gcloud iam service-accounts create solar-ai-pipeline \
     --display-name="Solar AI Pipeline Service Account"
   ```

2. **Grant required permissions**:
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:solar-ai-pipeline@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/bigquery.dataEditor"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:solar-ai-pipeline@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.objectAdmin"
   ```

3. **Create and download the service account key**:
   ```bash
   gcloud iam service-accounts keys create service-account-key.json \
     --iam-account=solar-ai-pipeline@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

4. **Add the key content** to GitHub Secrets as `GCP_SA_KEY`

### 3. Repository Configuration

1. **Enable GitHub Actions** in your repository settings
2. **Set up branch protection** for the `main` branch (recommended)
3. **Enable dependabot** for automated dependency updates

## 📅 Schedule & Triggers

### Automated Schedule
- **Daily at 6:00 AM UTC** - Full data collection and training pipeline
- **Runs on Ubuntu latest** with 3-hour timeout

### Manual Triggers
You can manually trigger the workflow with custom parameters:

```yaml
# Via GitHub UI: Actions → Solar AI Pipeline → Run workflow
force_retrain: true          # Force retraining even without new data
data_collection_hours: "48"  # Hours of recent data to collect
```

### Command Line Trigger
```bash
# Using GitHub CLI
gh workflow run solar-ai-pipeline.yml \
  -f force_retrain=true \
  -f data_collection_hours=24
```

## 🎯 Performance Thresholds

The pipeline uses these quality gates:

- **Minimum images for training**: 100 images total
- **Model accuracy threshold**: 80% mAP@0.5 for production deployment
- **Training timeout**: 3 hours maximum
- **Data collection timeout**: 2 hours maximum

## 📊 Monitoring & Artifacts

### Job Outputs
Each run provides:
- **Training logs** and metrics
- **Model validation reports** 
- **Data collection summaries**
- **Performance comparisons**

### Artifacts Retention
- **Training artifacts**: 30 days
- **Model files**: Permanent (via git)
- **Logs**: Available in workflow runs

### Release Management
Successful deployments automatically create:
- **Tagged releases** with model versions
- **Performance metrics** in release notes
- **Production model symlinks**

## 🔧 Customization

### Modify Training Parameters

Edit `.github/workflows/solar-ai-pipeline.yml`:

```yaml
# Training configuration
--epochs 100          # Increase training epochs
--batch_size 32       # Larger batch size for faster training
--patience 15         # Early stopping patience
```

### Adjust Data Collection

```yaml
# Data collection parameters
--max_images 1000     # Collect more images per run
--hours_lookback 48   # Look back further for storm events
```

### Change Schedule

```yaml
schedule:
  - cron: '0 */6 * * *'  # Run every 6 hours instead of daily
```

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Failures**
   ```
   Error: google.auth.exceptions.DefaultCredentialsError
   ```
   - Verify `GCP_SA_KEY` secret is correctly set
   - Check service account permissions

2. **Training Timeouts**
   ```
   Error: The operation was canceled
   ```
   - Reduce `--epochs` or `--max_images`
   - Use smaller `--batch_size`

3. **Insufficient Data**
   ```
   Warning: Insufficient data for retraining
   ```
   - Lower minimum image threshold
   - Use `force_retrain=true` for testing

### Debug Mode

Enable detailed logging by adding to workflow:

```yaml
env:
  PYTHONPATH: .
  ULTRALYTICS_VERBOSE: true
  DEBUG: true
```

### Manual Recovery

If the pipeline fails, you can recover manually:

```bash
# Clone repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Run data collection manually
python scripts/local_data_collector.py --max_images 200

# Train model manually  
python scripts/enhanced_distinct_yolo_trainer.py

# Test model
python scripts/comprehensive_solar_tester.py --model_path models/latest/best.pt
```

## 📈 Performance Optimization

### For Faster Training
1. **Use GPU runners** (GitHub-hosted or self-hosted)
2. **Optimize batch size** based on available memory
3. **Use mixed precision** training with `--amp`
4. **Enable early stopping** with appropriate patience

### For Better Accuracy
1. **Increase dataset size** with more collection hours
2. **Use data augmentation** during training
3. **Ensemble multiple models** for production
4. **Fine-tune hyperparameters** based on validation results

## 🌟 Best Practices

1. **Monitor resource usage** to avoid quota limits
2. **Test workflow changes** on feature branches first  
3. **Keep model artifacts** for rollback capability
4. **Review training logs** regularly for quality issues
5. **Set up alerting** for failed workflows
6. **Document model versions** with clear release notes

---

## 🆘 Support

For issues with the automated pipeline:

1. **Check workflow logs** in the Actions tab
2. **Review model validation reports** in artifacts
3. **Test components locally** before deployment
4. **Open issues** with detailed error information

The pipeline is designed to be robust and self-healing, automatically handling common issues like temporary service outages and data quality problems.