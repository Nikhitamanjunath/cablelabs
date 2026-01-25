#!/usr/bin/env python3
"""
SpecMon Web Scraper

A simple web scraper that automates login to the SpecMon website
using credentials from a YAML configuration file.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

import yaml
from pydantic import BaseModel, Field, field_validator, ValidationError
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from PIL import Image


class Credentials(BaseModel):
    """Credentials configuration."""
    username: str = Field(..., description="SpecMon username")
    password: str = Field(..., description="SpecMon password")


class Options(BaseModel):
    """Scraper options."""
    remember_me: bool = Field(default=False, description="Remember me checkbox")
    headless: bool = Field(default=True, description="Run browser in headless mode")


class GraphParameters(BaseModel):
    """Graph parameters configuration."""
    from_date: str = Field(..., description="From Date (e.g., '2024-01-01')")
    to_date: str = Field(..., description="To Date (e.g., '2024-01-31')")
    start_freq: str = Field(..., description="Start frequency (e.g., '3550')")
    end_freq: str = Field(..., description="End frequency (e.g., '3700')")
    power_threshold: str = Field(
        ..., 
        description="Power threshold: '-72 dBm/20MHz (Wi-Fi PD)' or '-89 dBm/MHz (CBRS ESC)'"
    )
    graph_type: str = Field(
        ...,
        description="Graph type: 'Airtime Utilization', 'Hour of Day', 'Heatmap-mean', 'Heatmap-max', 'Min-max-mean Area', or 'Weekly-mean'"
    )
    
    @field_validator('power_threshold')
    @classmethod
    def validate_power_threshold(cls, v: str) -> str:
        """Validate power threshold option."""
        valid_options = [
            '-72 dBm/20MHz (Wi-Fi PD)',
            '-89 dBm/MHz (CBRS ESC)'
        ]
        if v not in valid_options:
            raise ValueError(f"Power threshold must be one of: {valid_options}")
        return v
    
    @field_validator('graph_type')
    @classmethod
    def validate_graph_type(cls, v: str) -> str:
        """Validate graph type option."""
        valid_options = [
            'Airtime Utilization',
            'Hour of Day',
            'Heatmap-mean',
            'Heatmap-max',
            'Min-max-mean Area',
            'Weekly-mean'
        ]
        if v not in valid_options:
            raise ValueError(f"Graph type must be one of: {valid_options}")
        return v


class SpecMonConfig(BaseModel):
    """SpecMon configuration."""
    url: str = Field(..., description="SpecMon website URL")
    credentials: Credentials
    location: str = Field(..., description="Lab name to select on map (e.g., 'CableLabs, Louisville')")
    graph_parameters: GraphParameters = Field(..., description="Graph parameters for data visualization")
    options: Options = Field(default_factory=Options)
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate location is not empty."""
        if not v or not v.strip():
            raise ValueError("Location must be a non-empty string (e.g., 'CableLabs, Louisville')")
        return v.strip()


class Config(BaseModel):
    """Root configuration model."""
    specmon: SpecMonConfig


def load_config(config_path: str = None) -> Config:
    """
    Load and validate YAML configuration file using Pydantic.
    
    Args:
        config_path: Path to the configuration file (defaults to config.yaml in script directory)
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
        ValidationError: If configuration validation fails
    """
    if config_path is None:
        # Default to config.yaml in the same directory as this script
        config_path = Path(__file__).parent / "config.yaml"
    config_file = Path(config_path)
    
    if not config_file.exists():
        example_path = Path(__file__).parent / "config.yaml.example"
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found.\n"
            f"Please create '{config_path}' from '{example_path}' and fill in your credentials."
        )
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing YAML configuration file '{config_path}': {e}\n"
            f"Please check the file format and try again."
        ) from e
    
    if not config_data:
        raise ValueError(f"Configuration file '{config_path}' is empty.")
    
    # Pydantic automatically validates all fields and provides clear error messages
    return Config(**config_data)


class SpecMonScraper:
    """
    Main scraper class for automating SpecMon website interactions.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the scraper with configuration.
        
        Args:
            config: Configuration object from load_config()
        """
        self.config = config.specmon
        self.url = self.config.url
        self.credentials = self.config.credentials
        self.location = self.config.location
        self.graph_params = self.config.graph_parameters
        self.options = self.config.options
        
        self.playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None
    
    async def start_browser(self):
        """
        Launch browser with configured options.
        
        Raises:
            PlaywrightError: If browser launch fails
        """
        self.playwright = await async_playwright().start()
        
        headless = self.options.headless
        
        self.browser = await self.playwright.chromium.launch(
            headless=headless
        )
        
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        
        # Set default timeout (balanced for speed and network resilience)
        self.page.set_default_timeout(20000)  # 20 seconds
    
    async def close(self):
        """Clean up browser resources."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def login(self) -> bool:
        """
        Perform login to SpecMon website.
        
        Returns:
            True if login successful, False otherwise
            
        Raises:
            PlaywrightTimeoutError: If login form elements are not found
            PlaywrightError: If navigation or form submission fails
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start_browser() first.")
        
        print(f"Navigating to {self.url}...")
        # Retry navigation up to 3 times for network resilience
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                await self.page.goto(self.url, wait_until="domcontentloaded", timeout=30000)
                break
            except PlaywrightError as e:
                if attempt < max_retries:
                    print(f"  Navigation attempt {attempt} failed, retrying... ({e})")
                    await asyncio.sleep(2)  # Brief wait before retry
                else:
                    raise PlaywrightError(
                        f"Failed to navigate to {self.url} after {max_retries} attempts: {e}\n"
                        f"Please check your internet connection and the URL."
                    ) from e
        
        print("Waiting for login form...")
        
        username_field = await self._find_field([
            'input[type="email"]',
            'input[name="username"]',
            'input[name="email"]',
            'input[id*="user"]',
            'input[id*="email"]',
        ], timeout=2000)
        if not username_field:
            raise PlaywrightTimeoutError("Could not find username/email input field.")
        
        password_field = await self._find_field(['input[type="password"]'], timeout=2000)
        if not password_field:
            raise PlaywrightTimeoutError("Could not find password input field.")
        
        submit_button = await self._find_field([
            'button[type="submit"]',
            'input[type="submit"]',
            'button:has-text("Login")',
            'button:has-text("Sign in")',
        ], timeout=2000)
        if not submit_button:
            raise PlaywrightTimeoutError("Could not find login/submit button.")
        
        print("Filling login form...")
        
        # Fill username
        await username_field.fill(self.credentials.username)
        
        # Fill password
        await password_field.fill(self.credentials.password)
        
        # Handle "Remember me" checkbox if option is set
        if self.options.remember_me:
            remember_me_selectors = [
                'input[type="checkbox"][name*="remember" i]',
                'input[type="checkbox"][id*="remember" i]',
            ]
            
            for selector in remember_me_selectors:
                try:
                    checkbox = await self.page.query_selector(selector)
                    if checkbox:
                        await checkbox.check()
                        break
                except Exception:
                    continue
        
        print("Submitting login form...")
        
        # Get current URL before submission
        url_before = self.page.url
        
        # Submit form
        try:
            await submit_button.click()
        except PlaywrightError as e:
            raise PlaywrightError(f"Failed to submit login form: {e}") from e
        
        print("Waiting for login to complete...")
        # Wait for navigation to complete (more efficient than full page load)
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except PlaywrightTimeoutError:
            # Fallback to networkidle if domcontentloaded is too fast
            try:
                await self.page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeoutError:
                # If both fail, just wait a bit and continue (network might be slow)
                print("  Network may be slow, continuing anyway...")
                await asyncio.sleep(1)
        
        # Verify login success
        url_after = self.page.url
        
        # Check if URL changed (indicates redirect after successful login)
        if url_after != url_before and url_after != self.url:
            print("✓ Login successful! (Detected URL redirect)")
            return True
        
        # Check for error messages
        error_selectors = [
            '.error',
            '.alert-danger',
            '[class*="error" i]',
            '[id*="error" i]',
            'text=/invalid.*credential/i',
            'text=/incorrect.*password/i',
            'text=/login.*failed/i',
        ]
        
        for selector in error_selectors:
            try:
                error_element = await self.page.query_selector(selector)
                if error_element:
                    error_text = await error_element.inner_text()
                    if error_text and len(error_text.strip()) > 0:
                        print(f"✗ Login failed: {error_text.strip()}")
                        return False
            except Exception:
                continue
        
        # If no error found and we're still on the same page, assume failure
        if url_after == url_before:
            print("✗ Login may have failed. No redirect detected and no error message found.")
            print("Please check your credentials and try again.")
            return False
        
        # If we got here and URL changed, assume success
        print("✓ Login successful!")
        return True
    
    async def _find_field(self, selectors: list[str], timeout: int = 1000) -> any:
        """
        Helper to find field using multiple selectors efficiently.
        Uses shorter individual timeouts to try selectors faster.
        """
        # Use shorter timeout per selector to fail fast and try next one
        per_selector_timeout = min(500, timeout // len(selectors)) if selectors else timeout
        per_selector_timeout = max(200, per_selector_timeout)  # Minimum 200ms
        
        for selector in selectors:
            try:
                # Try query_selector first (instant check)
                field = await self.page.query_selector(selector)
                if field:
                    # Verify it's visible
                    is_visible = await field.is_visible()
                    if is_visible:
                        return field
                
                # If not found, wait briefly for it to appear
                field = await self.page.wait_for_selector(selector, timeout=per_selector_timeout, state="visible")
                if field:
                    return field
            except PlaywrightTimeoutError:
                continue
        return None
    
    async def select_location(self) -> bool:
        """Select a lab location on the map after login by finding the Mapbox marker."""
        if not self.page:
            raise RuntimeError("Browser not started. Call start_browser() first.")
        
        print(f"Looking for location: {self.location}...")
        
        # Wait for markers with reasonable timeout for network resilience
        await self.page.wait_for_selector('.mapboxgl-marker', timeout=8000, state="visible")
        markers = await self.page.query_selector_all('.mapboxgl-marker')
        
        if not markers:
            raise PlaywrightTimeoutError("No map markers found on the page.")
        
        print(f"Found {len(markers)} marker(s) on the map")
        
        location_lower = self.location.lower()
        for marker in markers:
            try:
                # Click without waiting (faster)
                await marker.click(no_wait_after=True)
                # Minimal wait for popup/text to appear
                await self.page.wait_for_timeout(150)
                page_text = await self.page.evaluate("() => document.body.innerText || ''")
                if location_lower in page_text.lower():
                    print(f"Found location '{self.location}' by clicking marker")
                    print(f"✓ Successfully selected {self.location}")
                    return True
            except Exception:
                continue
        
        raise PlaywrightTimeoutError(
            f"Could not find lab location '{self.location}' on the map.\n"
            f"Searched through {len(markers)} marker(s) but none matched."
        )
    
    async def _select_option(self, selectors: list[str], value: str, field_name: str) -> None:
        """Helper to select radio/select option efficiently."""
        field = await self._find_field(selectors, timeout=1500)
        if field:
            tag = await field.evaluate("el => el.tagName.toLowerCase()")
            if tag == 'select':
                await field.select_option(label=value)
            else:
                await field.click()
            print(f"  ✓ Selected {field_name}: {value}")
            return
        
        label = await self.page.query_selector(f'label:has-text("{value}")')
        if label:
            await label.click()
            print(f"  ✓ Selected {field_name}: {value}")
            return
        
        raise PlaywrightTimeoutError(f"Could not find '{field_name}' field")
    
    async def fill_graph_parameters(self, date: str) -> bool:
        """
        Fill in graph parameters and submit the form.
        
        Args:
            date: The date to use for both from_date and to_date fields (YYYY-MM-DD format)
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start_browser() first.")
        
        print(f"Filling graph parameters for date: {date}...")
        
        # Fill date fields - use the same date for both from and to
        from_date_field = await self._find_field([
            'input[name*="from" i][type*="date" i]',
            'input[id*="from" i][type*="date" i]',
            'input[type="date"]:first-of-type',
        ], timeout=1500)
        if not from_date_field:
            raise PlaywrightTimeoutError("Could not find 'From Date' field")
        await from_date_field.fill(date)
        print(f"  ✓ Filled From Date: {date}")
        
        to_date_field = await self._find_field([
            'input[name*="to" i][type*="date" i]',
            'input[id*="to" i][type*="date" i]',
            'input[type="date"]:last-of-type',
        ], timeout=1500)
        if not to_date_field:
            raise PlaywrightTimeoutError("Could not find 'To Date' field")
        await to_date_field.fill(date)
        print(f"  ✓ Filled To Date: {date}")
        
        # Fill frequency fields
        start_freq = await self._find_field([
            'input[name*="start" i][name*="freq" i]',
            'input[id*="start" i][id*="freq" i]',
            'input[type="number"]:first-of-type',
        ], timeout=1500)
        if not start_freq:
            raise PlaywrightTimeoutError("Could not find 'Start freq' field")
        await start_freq.fill(self.graph_params.start_freq)
        print(f"  ✓ Filled Start freq: {self.graph_params.start_freq}")
        
        end_freq = await self._find_field([
            'input[name*="end" i][name*="freq" i]',
            'input[id*="end" i][id*="freq" i]',
            'input[type="number"]:last-of-type',
        ], timeout=1500)
        if not end_freq:
            raise PlaywrightTimeoutError("Could not find 'End freq' field")
        await end_freq.fill(self.graph_params.end_freq)
        print(f"  ✓ Filled End freq: {self.graph_params.end_freq}")
        
        # Select options
        await self._select_option([
            f'input[type="radio"][value*="{self.graph_params.power_threshold}"]',
            f'input[type="radio"][value*="-72"]',
            f'input[type="radio"][value*="-89"]',
            'select[name*="power" i]',
        ], self.graph_params.power_threshold, "Power threshold")
        
        await self._select_option([
            f'input[type="radio"][value*="{self.graph_params.graph_type}"]',
            'select[name*="graph" i]',
            'select[name*="type" i]',
        ], self.graph_params.graph_type, "Graph type")
        
        # Submit
        submit = await self._find_field([
            'button[type="submit"]',
            'button:has-text("Submit")',
            'button:has-text("Generate")',
            'input[type="submit"]',
            'button[class*="submit" i]',
            'button[id*="submit" i]',
        ], timeout=1500)
        if not submit:
            raise PlaywrightTimeoutError("Could not find submit button")
        
        print("Submitting form...")
        # Click without waiting for navigation (faster)
        await submit.click(no_wait_after=True)
        print("✓ Form submitted successfully")
        return True
    
    async def capture_graph_image(self, date: str) -> bool:
        """
        Wait for graph to appear and capture it as an image.
        
        Args:
            date: The date string (YYYY-MM-DD) to use for the filename
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start_browser() first.")
        
        print("Waiting for graph to load...")
        
        # Wait for any SVG element with class "main-svg" to appear
        try:
            await self.page.wait_for_selector(
                'svg.main-svg', timeout=15000, state="visible"
            )
        except PlaywrightTimeoutError:
            raise PlaywrightTimeoutError("Graph SVG did not appear after form submission.")
        
        # Scroll to the bottom of the page first (no wait needed, scroll is instant)
        await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        
        # Find all SVG elements with class "main-svg" and get the last one (at the bottom)
        svg_elements = await self.page.query_selector_all('svg.main-svg')
        if not svg_elements:
            raise PlaywrightTimeoutError("Could not find SVG graph element.")
        
        # Get the last SVG element (the one at the bottom)
        svg_element = svg_elements[-1]
        
        # Scroll to the specific SVG element to ensure it's in view (no wait needed)
        await svg_element.scroll_into_view_if_needed()
        
        # Create data/preprocessed directory if it doesn't exist
        pre_processed_dir = Path("data") / "preprocessed"
        pre_processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from date
        filename = date.replace("-", "_") + ".png"
        pre_processed_filepath = pre_processed_dir / filename
        
        print(f"Capturing graph image (pre-processed) as {pre_processed_filepath}...")
        
        # Screenshot the SVG element directly
        try:
            await svg_element.screenshot(path=str(pre_processed_filepath))
            print(f"✓ Pre-processed graph image saved to {pre_processed_filepath}")
            return True
        except Exception as e:
            # Fallback: screenshot using the SVG's bounding box
            try:
                box = await svg_element.bounding_box()
                if box:
                    await self.page.screenshot(
                        path=str(pre_processed_filepath),
                        clip={
                            "x": max(0, box["x"]),
                            "y": max(0, box["y"]),
                            "width": box["width"],
                            "height": box["height"]
                        }
                    )
                    print(f"✓ Pre-processed graph image saved to {pre_processed_filepath}")
                    return True
            except Exception:
                raise PlaywrightError(f"Failed to capture graph image: {e}")
    
    async def click_back_button(self) -> bool:
        """Click the back button to return to the form page."""
        if not self.page:
            raise RuntimeError("Browser not started. Call start_browser() first.")
        
        print("Clicking back button...")
        
        # Try to find the back button with various selectors (faster timeout)
        back_button = await self._find_field([
            'button:has-text("Back")',
            'a:has-text("Back")',
            'button[class*="back" i]',
            'a[class*="back" i]',
            'button[id*="back" i]',
            'a[id*="back" i]',
            'button:has-text("←")',
            'a:has-text("←")',
        ], timeout=1500)
        
        if not back_button:
            # Try browser back button as fallback (faster)
            print("  Back button not found, using browser back navigation...")
            await self.page.go_back(wait_until="domcontentloaded", timeout=5000)
            print("✓ Navigated back")
            return True
        
        # Click without waiting, then wait for navigation separately (faster)
        await back_button.click(no_wait_after=True)
        # Wait for navigation to start (non-blocking wait)
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except PlaywrightTimeoutError:
            # If network is slow, wait a bit and continue
            print("  Network may be slow, waiting a bit longer...")
            await asyncio.sleep(1)
        print("✓ Back button clicked")
        return True
    
    def crop_image(self, pre_processed_path: Path, output_path: Path) -> bool:
        """
        Crop the captured image to show only the central plot area.
        
        Args:
            pre_processed_path: Path to the pre-processed (uncropped) image
            output_path: Path where the cropped image should be saved
        
        Uses fixed coordinates based on the rect.nsewdrag.drag element:
        - x="175" y="100" width="733" height="564"
        - Crop box: (175, 100) to (908, 664)
        """
        if not pre_processed_path.exists():
            raise FileNotFoundError(f"Pre-processed image file not found: {pre_processed_path}")
        
        # Fixed crop coordinates based on the plot area rect element
        # These match the rect.nsewdrag.drag element: x=175, y=100, width=733, height=564
        left = 175
        top = 100
        right = 908  # 175 + 733
        bottom = 664  # 100 + 564
        
        print(f"Cropping image to plot area: ({left}, {top}) to ({right}, {bottom})...")
        
        try:
            # Open the pre-processed image
            img = Image.open(pre_processed_path)
            
            # Validate coordinates are within image bounds
            width, height = img.size
            left = max(0, min(left, width))
            top = max(0, min(top, height))
            right = max(left, min(right, width))
            bottom = max(top, min(bottom, height))
            
            # Crop the image using PIL's crop method: (left, top, right, bottom)
            cropped_img = img.crop((left, top, right, bottom))
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(exist_ok=True)
            
            # Save the cropped image to the output path
            cropped_img.save(output_path)
            print(f"✓ Image cropped and saved to {output_path}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to crop image: {e}")


def generate_date_range(from_date: str, to_date: str) -> list[str]:
    """
    Generate a list of dates from from_date to to_date (inclusive).
    
    Args:
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
    
    Returns:
        List of date strings in YYYY-MM-DD format
    """
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    
    if start > end:
        raise ValueError(f"from_date ({from_date}) must be <= to_date ({to_date})")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates


async def main():
    """
    Main entry point for the scraper.
    """
    try:
        print("Loading config...")
        config = load_config()
        print("✓ Configuration loaded successfully")
        
        print("Initializing scraper...")
        scraper = SpecMonScraper(config)
        
        print("Opening browser...")
        await scraper.start_browser()
        
        try:
            success = await scraper.login()
            
            if success:
                print("\n✓ Login completed successfully!")
                
                # Select location on the map
                try:
                    await scraper.select_location()
                    print("\n✓ Location selection completed successfully!")
                except PlaywrightTimeoutError as e:
                    print(f"\n✗ Location Selection Error: {e}", file=sys.stderr)
                    sys.exit(1)
                except PlaywrightError as e:
                    print(f"\n✗ Location Selection Error: {e}", file=sys.stderr)
                    sys.exit(1)
                
                # Generate date range from from_date to to_date (inclusive)
                dates = generate_date_range(
                    scraper.graph_params.from_date,
                    scraper.graph_params.to_date
                )
                print(f"\nProcessing {len(dates)} date(s): {dates[0]} to {dates[-1]}")
                
                # Process each date
                for i, date in enumerate(dates, 1):
                    print(f"\n{'='*60}")
                    print(f"Processing date {i}/{len(dates)}: {date}")
                    print(f"{'='*60}")
                    
                    try:
                        # Fill graph parameters and submit (using same date for both fields)
                        await scraper.fill_graph_parameters(date)
                        print(f"\n✓ Graph parameters submitted successfully for {date}!")
                    except PlaywrightTimeoutError as e:
                        print(f"\n✗ Graph Parameters Error for {date}: {e}", file=sys.stderr)
                        continue  # Continue to next date
                    except PlaywrightError as e:
                        print(f"\n✗ Graph Parameters Error for {date}: {e}", file=sys.stderr)
                        continue  # Continue to next date
                    
                    # Capture graph image
                    try:
                        await scraper.capture_graph_image(date)
                        print(f"\n✓ Graph image captured successfully for {date}!")
                        
                        # Crop the image to show only the plot area
                        filename = date.replace("-", "_") + ".png"
                        pre_processed_dir = Path("data") / "preprocessed"
                        images_dir = Path("data") / "images"
                        pre_processed_path = pre_processed_dir / filename
                        output_path = images_dir / filename
                        
                        scraper.crop_image(pre_processed_path, output_path)
                        print(f"\n✓ Image processing completed successfully for {date}!")
                    except PlaywrightTimeoutError as e:
                        print(f"\n✗ Graph Capture Error for {date}: {e}", file=sys.stderr)
                        continue  # Continue to next date
                    except PlaywrightError as e:
                        print(f"\n✗ Graph Capture Error for {date}: {e}", file=sys.stderr)
                        continue  # Continue to next date
                    except Exception as e:
                        print(f"\n✗ Image Processing Error for {date}: {e}", file=sys.stderr)
                        continue  # Continue to next date
                    
                    # Click back button to return to form (except for last date)
                    if i < len(dates):
                        try:
                            await scraper.click_back_button()
                            print(f"\n✓ Returned to form page")
                            
                            # Select location again after going back
                            try:
                                await scraper.select_location()
                                print(f"\n✓ Location reselected successfully!")
                            except PlaywrightTimeoutError as e:
                                print(f"\n✗ Location Selection Error after back: {e}", file=sys.stderr)
                                continue  # Skip to next date if location selection fails
                            except PlaywrightError as e:
                                print(f"\n✗ Location Selection Error after back: {e}", file=sys.stderr)
                                continue  # Skip to next date if location selection fails
                        except Exception as e:
                            print(f"\n✗ Back Button Error: {e}", file=sys.stderr)
                            # Try to continue anyway
                            continue
                
                print(f"\n{'='*60}")
                print(f"✓ Completed processing all {len(dates)} date(s)")
                print(f"{'='*60}")
                sys.exit(0)
            else:
                print("\n✗ Login failed. Please check your credentials.")
                sys.exit(1)
        
        finally:
            print("Closing browser...")
            await scraper.close()
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except yaml.YAMLError as e:
        print(f"\n✗ Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except ValidationError as e:
        print(f"\n✗ Configuration Validation Error:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    
    except PlaywrightTimeoutError as e:
        print(f"\n✗ Timeout Error: {e}", file=sys.stderr)
        print("This might indicate the website structure has changed or the page is slow to load.", file=sys.stderr)
        sys.exit(1)
    
    except PlaywrightError as e:
        print(f"\n✗ Browser Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
