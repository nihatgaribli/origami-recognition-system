# Origami Notebook

A hybrid origami classification and information system combining AI model inference with database querying and web scraping capabilities.

## Features

- **AI Model**: MobileNetV2-based Transfer Learning for origami classification
- **Hybrid Assistant**: GUI combining image analysis with origami database queries
- **Web Scrapers**: Data collection from ORIWiki, CFC (Craft and Craft Origami), and ORC (Origami Resource Center)
- **Database**: PostgreSQL backend for model metadata and creator information
- **Image Storage**: Cloudinary integration for scalable image hosting

## Project Structure

```
origami_notebook/
├── ai/                          # Model training and inference
│   ├── train_model.py          # MobileNetV2 training pipeline
│   ├── predict_image.py        # Single image inference
│   ├── hybrid_origami_assistant.py  # GUI application
│   ├── data_generator.py       # Custom Keras data generator
│   └── image_preprocessing.py  # Image normalization
├── scrapers/                   # Web scraping modules
│   ├── comprehensive_scraper.py  # ORIWiki scraping
│   ├── cfc_scraping.py         # CFC website scraping
│   ├── orc_scraping.py         # ORC website scraping
│   └── cfc_download_images.py  # Image upload to Cloudinary
├── visualization/              # Analytics and reporting
│   ├── models_stats.py         # Model statistics visualization
│   ├── advanced_charts.py      # Advanced analytics
│   └── _db_config.py           # Database configuration
├── db_tools/                   # Database utilities
│   ├── test_db_connection.py   # Connection testing
│   └── _clear_database.py      # Data cleanup
├── pipelines/                  # Orchestration scripts
│   └── rebuild_ai_pipeline.py  # Full pipeline execution
└── W_database/                 # Database exports
    └── origami_full_db.sql     # Database schema
```

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/origami_notebook.git
cd origami_notebook
```

### 2. Create Environment Variables
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```
POSTGRES_HOST=your_host
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Database
```bash
python db_tools/test_db_connection.py
```

## Usage

### Train Model
```bash
python ai/train_model.py
```

### Run GUI Assistant
```bash
python ai/hybrid_origami_assistant.py
```

### Scrape Data
```bash
# Full scrape (ORIWiki, CFC, ORC)
python scrapers/full_scrape_direct.py

# Individual sources
python scrapers/comprehensive_scraper.py
python scrapers/cfc_scraping.py
python scrapers/orc_scraping.py
```

### Generate Reports
```bash
python visualization/models_stats.py
python visualization/advanced_charts.py
```

## Configuration

Environment variables are read from `.env` file. Key variables:

- `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD` - Database connection
- `CLOUDINARY_*` - Image hosting credentials
- `ORIGAMI_EPOCHS` - Training epochs (default: 50)
- `ORIGAMI_MAX_CLASSES` - Max origami categories (default: 50)

## Dataset

The AI model trains on origami models from:
- **ORIWiki**: 2000+ models
- **CFC**: Diagrams and instructional materials
- **ORC**: Community shared models and resources

## Model Architecture

- **Base**: MobileNetV2 (ImageNet weights)
- **Preprocessing**: 224×224 resize, normalization to [0, 1]
- **Data Augmentation**: Random rotation, zoom, and flip
- **Output**: Softmax classification across origami categories

## License

This project is provided as-is for educational purposes.

## Contributing

Contributions welcome! Please follow the code style and add comments for complex logic.

