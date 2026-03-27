# Origami Notebook

A hybrid origami classification and information system combining AI model inference, large language models (LLM), and database querying capabilities. Features professional-grade image analysis with Groq LLM integration.

## 🎯 Key Features

- **🤖 Groq LLM Integration** (llama-3.3-70b-versatile)
  - AI-powered natural language query understanding
  - Professional image analysis with structured 5-section output
  - Intelligent confidence assessment and technical specifications
  - Expert recommendations and geometric analysis

- **📸 AI Image Classification**: MobileNetV2-based Transfer Learning for origami model recognition
  - Top-3 predictions with confidence scoring
  - Geometric and structural analysis
  - Automatic difficulty level assessment

- **🎨 Hybrid GUI Assistant** (CustomTkinter)
  - Real-time image upload and analysis
  - Natural language query support
  - Database-backed origami model information
  - Professional Markdown output formatting

- **🌐 Web Scrapers**: Multi-source data collection
  - ORIWiki (comprehensive_scraper.py)
  - CFC - Craft and Craft Origami (cfc_scraping.py)
  - ORC - Origami Resource Center (orc_scraping.py)

- **💾 PostgreSQL Database**: 49,885+ origami models with creator info, difficulty levels, material requirements, and tutorial links

- **☁️ Cloudinary Integration**: Scalable image hosting and optimization

## 🧠 AI Analysis Features

### Image Analysis Output
When you upload an origami image, the system provides:

1. **Confidence Assessment** - Visual indicators (✅ High / 🟡 Moderate / ⚠️ Low)
2. **Top-3 Predictions** - Model names with confidence percentages and geometric reasoning
3. **Geometric Analysis** - Expert commentary on fold structure, symmetry, and mechanics
4. **Technical Specifications** - Difficulty level, creator info, material requirements
5. **Expert Recommendation** - Motivating insights for the user

### Natural Language Query Processing
The Groq LLM intelligently parses user queries to:
- Extract origami names and creator information
- Identify difficulty filters (Beginner/Intermediate/Advanced)
- Detect material requirements (cutting, glue)
- Support category-based searches (Animals, Flowers, etc.)
- Return natural language responses with relevant results

### Example Queries
```
"Show me all origami models created by Ilan Garibi"
"Find 3 easy origami animals without cutting"
"What's the difficulty of a crane?"
"Recommend intermediate butterfly models"
```

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

### Run GUI Assistant (Main Application)
```bash
python ai/hybrid_origami_assistant.py
```
**Features:**
- Upload origami images for AI-powered recognition
- Ask natural language questions about origami models
- Get professional analysis with confidence scores, geometric reasoning, and expert recommendations
- Database lookup with difficulty ratings and creator information

### Configuration
Before running, ensure your `.env` file is configured with:
```bash
# Groq LLM API
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Database credentials
POSTGRES_HOST=your_host
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# Cloudinary (for image uploads)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

### Train Model
```bash
python ai/train_model.py
```

### Single Image Prediction
```bash
python ai/predict_image.py path/to/image.jpg
```

### Scrape Data
```bash
# Full scrape from all sources
python scrapers/full_scrape_direct.py

# Individual sources
python scrapers/comprehensive_scraper.py  # ORIWiki
python scrapers/cfc_scraping.py          # CFC
python scrapers/orc_scraping.py          # ORC
```

### Generate Visualizations
```bash
python visualization/models_stats.py
python visualization/advanced_charts.py
```


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

## 🛠️ Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **LLM** | Groq API | llama-3.3-70b-versatile model |
| **Image Classification** | TensorFlow/Keras | MobileNetV2 transfer learning |
| **GUI Framework** | CustomTkinter | Dark theme, real-time updates |
| **Database** | PostgreSQL | 49,885+ model records |
| **Image Hosting** | Cloudinary | Optimized image delivery |
| **Web Scraping** | BeautifulSoup4 | Multi-source data collection |
| **Data Processing** | Pandas/NumPy | Analytics and transformation |

## 🔐 Security Notes

- All sensitive credentials stored in `.env` (not committed)
- Environment variables for database and API keys
- No hardcoded secrets in repository
- Template provided: `.env.example`

## 📊 Performance

- **Model Inference**: ~2-5 seconds per image (GPU: <1s)
- **Database Query**: <100ms average
- **Groq LLM Response**: 1-3 seconds (including API latency)
- **GUI Responsiveness**: Multiprocessing prevents UI freeze

## 🚀 Deployment

Ready for cloud deployment:
- Containerizable (Docker support)
- PostgreSQL-compatible deployment environments
- Groq API available globally
- Cloudinary CDN for image delivery

## 📝 License

Educational and research purposes.

## 👨‍💻 Author

**Nihat Garibli** - GitHub: [@nihatgaribli](https://github.com/nihatgaribli)

**Repository**: [origami-recognition-system](https://github.com/nihatgaribli/origami-recognition-system)

---

*Last Updated: March 2026 | Groq LLM Integration v1.0*

