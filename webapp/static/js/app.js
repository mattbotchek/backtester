/* Stock Predictor — Client-side JavaScript */

// Ticker search functionality
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('ticker-search');
    const searchResults = document.getElementById('search-results');

    if (!searchInput || !searchResults) return;

    let debounceTimer;

    searchInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const q = searchInput.value.trim();

        if (q.length < 1) {
            searchResults.classList.remove('show');
            return;
        }

        debounceTimer = setTimeout(() => {
            fetch(`/api/tickers?q=${encodeURIComponent(q)}`)
                .then(r => r.json())
                .then(data => {
                    const tickers = data.tickers || [];
                    if (tickers.length === 0) {
                        searchResults.classList.remove('show');
                        return;
                    }
                    searchResults.innerHTML = tickers
                        .map(t => `<a href="/stock/${t}">${t}</a>`)
                        .join('');
                    searchResults.classList.add('show');
                });
        }, 200);
    });

    // Submit on enter
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const q = searchInput.value.trim().toUpperCase();
            if (q) window.location.href = `/stock/${q}`;
        }
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.nav-search')) {
            searchResults.classList.remove('show');
        }
    });
});
