/**
 * ML LabVerse — Main UI & Navigation Interactions
 * Soft Academic Futurism Theme
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Mobile Navigation Drawer Toggle
    const mobileMenuToggle = document.getElementById("mobileMenuToggle");
    const closeDrawerBtn = document.getElementById("closeDrawerBtn");
    const mobileDrawer = document.getElementById("mobileDrawer");
    const mobileNavOverlay = document.getElementById("mobileNavOverlay");

    function openMobileDrawer() {
        if (mobileDrawer) mobileDrawer.classList.add("open");
        if (mobileNavOverlay) mobileNavOverlay.classList.add("open");
        document.body.style.overflow = "hidden";
    }

    function closeMobileDrawer() {
        if (mobileDrawer) mobileDrawer.classList.remove("open");
        if (mobileNavOverlay) mobileNavOverlay.classList.remove("open");
        document.body.style.overflow = "";
    }

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener("click", openMobileDrawer);
    }

    if (closeDrawerBtn) {
        closeDrawerBtn.addEventListener("click", closeMobileDrawer);
    }

    if (mobileNavOverlay) {
        mobileNavOverlay.addEventListener("click", closeMobileDrawer);
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && mobileDrawer && mobileDrawer.classList.contains("open")) {
            closeMobileDrawer();
        }
    });

    // 2. Experiment Search Filtering (Home & Experiments page)
    const searchInputs = [
        document.getElementById("expSearchInput"),
        document.getElementById("experimentSearchInput")
    ].filter(Boolean);

    const expCards = document.querySelectorAll(".modern-exp-card, .exp-card");

    searchInputs.forEach(searchInput => {
        searchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            expCards.forEach(card => {
                const title = (card.querySelector(".card-title, .exp-card-title")?.textContent || "").toLowerCase();
                const algo = (card.querySelector(".card-algo-code, .exp-card-algo")?.textContent || "").toLowerCase();
                const desc = (card.querySelector(".card-summary, .exp-card-desc")?.textContent || "").toLowerCase();
                const num = (card.querySelector(".card-number-tag, .exp-number-badge")?.textContent || "").toLowerCase();

                if (!query || title.includes(query) || algo.includes(query) || desc.includes(query) || num.includes(query)) {
                    card.style.display = "flex";
                } else {
                    card.style.display = "none";
                }
            });
        });
    });

    // 3. Category Filter Tabs / Pills
    const filterPills = document.querySelectorAll(".filter-pill, .filter-tab");
    if (filterPills.length > 0 && expCards.length > 0) {
        filterPills.forEach(pill => {
            pill.addEventListener("click", () => {
                filterPills.forEach(p => p.classList.remove("active"));
                pill.classList.add("active");

                const filter = pill.dataset.filter;
                expCards.forEach(card => {
                    const cardCategory = card.dataset.category;
                    if (filter === "all" || cardCategory === filter) {
                        card.style.display = "flex";
                    } else {
                        card.style.display = "none";
                    }
                });
            });
        });
    }

    // 4. Number Counter Animation for Dashboard Stats
    const countEls = document.querySelectorAll(".stat-box-futurism .stat-big-num, .stat-info .num");
    countEls.forEach(el => {
        const text = el.textContent.trim();
        const target = parseInt(text, 10);
        if (!isNaN(target) && target > 0) {
            let current = 0;
            const step = Math.ceil(target / 15) || 1;
            const timer = setInterval(() => {
                current += step;
                if (current >= target) {
                    el.textContent = target;
                    clearInterval(timer);
                } else {
                    el.textContent = current;
                }
            }, 35);
        }
    });

    // 5. Initialize Lucide Icons if available
    if (window.lucide) {
        lucide.createIcons();
    }
});
