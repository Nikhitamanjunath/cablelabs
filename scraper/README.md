# SpecMon Web Scraper

A simple, easy-to-use web scraper that automates login to the SpecMon website (https://specmon.cablelabs.com/home) using credentials from a configuration file.

## Features

- Automated login using YAML configuration
- Headless or headed browser mode
- Extensible architecture for future features (screenshots, data extraction, etc.)
- Clear error handling and user feedback

## Setup

### 1. Create Virtual Environment

```bash
cd specmon-scraper
python3 -m venv .venv
```

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright Browsers

```bash
playwright install chromium
```

### 5. Create Configuration File

Copy the example configuration file and fill in your credentials:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your SpecMon username and password:

```yaml
specmon:
  url: "https://specmon.cablelabs.com/home"
  credentials:
    username: "your_username"
    password: "your_password"
  options:
    remember_me: false
    headless: true
```

**Important:** The `config.yaml` file is gitignored and will not be committed to version control.

## Usage

Run the scraper:

```bash
python scraper.py
```

The scraper will:
1. Load configuration from `config.yaml`
2. Open the SpecMon website
3. Fill in login credentials
4. Submit the login form
5. Verify successful login
6. Display login status

## Configuration Options

- `url`: The SpecMon website URL (default: https://specmon.cablelabs.com/home)
- `credentials.username`: Your SpecMon username
- `credentials.password`: Your SpecMon password
- `options.remember_me`: Whether to check the "Remember me" checkbox (default: false)
- `options.headless`: Run browser in headless mode (default: true)

## Error Handling

The scraper handles common errors gracefully:

- **Missing config file**: Clear message with instructions to create `config.yaml`
- **Invalid credentials**: Detects and reports authentication failures
- **Network errors**: Handles timeouts and connection issues
- **Missing form elements**: Reports if login form structure has changed

## Future Extensions

The code structure is designed to easily add:
- Screenshot capture functionality
- Page content saving
- Navigation to specific pages after login
- Data extraction from pages

## Requirements

- Python 3.8 or higher
- Playwright browser automation framework
- PyYAML for configuration file parsing
