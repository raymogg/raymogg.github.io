// Helper function to get cached data from localStorage
function getCachedData(key, maxAgeMinutes = 10) {
    const cached = localStorage.getItem(key);
    if (!cached) return null;

    try {
        const { data, timestamp } = JSON.parse(cached);
        const age = Date.now() - timestamp;
        const maxAge = maxAgeMinutes * 60 * 1000;

        if (age < maxAge) {
            return data;
        }
    } catch (error) {
        // Invalid cache data, ignore
    }

    return null;
}

// Helper function to set cached data in localStorage
function setCachedData(key, data) {
    const cacheObject = {
        data: data,
        timestamp: Date.now()
    };
    localStorage.setItem(key, JSON.stringify(cacheObject));
}

// Fetch Brisbane temperature using Open-Meteo API
async function fetchTemperature() {
    const cacheKey = 'brisbane-weather';

    // Try to get from cache first
    const cachedTemp = getCachedData(cacheKey, 10);
    if (cachedTemp !== null) {
        document.getElementById('temperature').textContent = `Brisbane: ${cachedTemp}`;
        return;
    }

    try {
        // Using Open-Meteo API (free, no API key required, more reliable)
        const response = await fetch('https://api.open-meteo.com/v1/forecast?latitude=-27.47&longitude=153.03&current=temperature_2m&temperature_unit=celsius');
        if (response.ok) {
            const data = await response.json();
            const temp = Math.round(data.current.temperature_2m);
            const displayText = `${temp}°C`;

            document.getElementById('temperature').textContent = `Brisbane: ${displayText}`;
            setCachedData(cacheKey, displayText);
        } else {
            document.getElementById('temperature').textContent = 'Brisbane: N/A';
        }
    } catch (error) {
        document.getElementById('temperature').textContent = 'Brisbane: N/A';
    }
}

// Fetch Ethereum price from CoinGecko
async function fetchEthPrice() {
    const cacheKey = 'eth-price';

    // Try to get from cache first
    const cachedPrice = getCachedData(cacheKey, 10);
    if (cachedPrice !== null) {
        document.getElementById('eth-price').textContent = `ETH: ${cachedPrice}`;
        return;
    }

    try {
        const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd');
        if (response.ok) {
            const data = await response.json();
            const price = data.ethereum.usd;
            const displayText = `$${price.toLocaleString()} USD`;

            document.getElementById('eth-price').textContent = `ETH: ${displayText}`;
            setCachedData(cacheKey, displayText);
        } else {
            document.getElementById('eth-price').textContent = 'ETH: N/A';
        }
    } catch (error) {
        document.getElementById('eth-price').textContent = 'ETH: N/A';
    }
}

// Load data on page load
fetchTemperature();
fetchEthPrice();

// Refresh every 5 minutes
setInterval(fetchTemperature, 5 * 60 * 1000);
setInterval(fetchEthPrice, 5 * 60 * 1000);
