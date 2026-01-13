// Fetch Brisbane temperature
async function fetchTemperature() {
    try {
        // Using wttr.in service (simple, no API key required)
        const response = await fetch('https://wttr.in/Brisbane?format=%t');
        if (response.ok) {
            const temp = await response.text();
            document.getElementById('temperature').textContent = `Brisbane: ${temp.trim()}`;
        } else {
            document.getElementById('temperature').textContent = 'Brisbane: N/A';
        }
    } catch (error) {
        document.getElementById('temperature').textContent = 'Brisbane: N/A';
    }
}

// Fetch Ethereum price from CoinGecko
async function fetchEthPrice() {
    try {
        const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd');
        if (response.ok) {
            const data = await response.json();
            const price = data.ethereum.usd;
            document.getElementById('eth-price').textContent = `ETH: $${price.toLocaleString()} USD`;
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
