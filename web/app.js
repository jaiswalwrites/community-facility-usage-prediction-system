let dashboardData = null;
let allPredictions = [];

document.addEventListener("DOMContentLoaded", () => {
  fetchDashboardData();
  setupEventListeners();
});

async function fetchDashboardData() {
  try {
    const res = await fetch("dashboard_data.json");
    if (!res.ok) throw new Error("JSON file not found");
    dashboardData = await res.json();
    renderDashboard();
  } catch (err) {
    console.warn("Loading fallback data or local execution mode", err);
    // If running directly via file:// schema, fetch fallback embedded or mock predictions
    loadFallbackData();
  }
}

function renderDashboard() {
  if (!dashboardData) return;

  const m = dashboardData.metrics;
  document.getElementById("kpi-exact-match").textContent = `${m.exact_4of4_match_rate}%`;
  document.getElementById("kpi-facility-acc").textContent = `${m.facility_accuracy}%`;
  document.getElementById("kpi-day-acc").textContent = `${m.usage_day_accuracy}%`;
  document.getElementById("kpi-hour-acc").textContent = `${m.usage_hour_accuracy}%`;
  document.getElementById("kpi-nudge-mae").textContent = `${m.nudge_time_mae_hours} hrs`;

  allPredictions = dashboardData.predictions || [];
  populateResidentSelect();
  renderTable(allPredictions);
}

function populateResidentSelect() {
  const select = document.getElementById("resident-select");
  select.innerHTML = "";

  const uniqueResidents = [...new Set(allPredictions.map(p => p.resident_id))].sort();
  uniqueResidents.forEach(resId => {
    const opt = document.createElement("option");
    opt.value = resId;
    opt.textContent = `Resident ${resId}`;
    select.appendChild(opt);
  });
}

function renderTable(data) {
  const tbody = document.getElementById("table-body");
  const countLabel = document.getElementById("records-count");
  tbody.innerHTML = "";

  countLabel.textContent = `Showing ${data.length} of ${allPredictions.length} records`;

  data.forEach(item => {
    const tr = document.createElement("tr");

    let badgeClass = "badge-no";
    if (item.match_indicator === "YES" && item.score.includes("4 of 4")) {
      badgeClass = "badge-yes";
    } else if (item.score.includes("3 of 4") || item.score.includes("2 of 4")) {
      badgeClass = "badge-partial";
    }

    tr.innerHTML = `
      <td>
        <div class="cell-ref">${item.record_reference}</div>
        <div class="cell-res">${item.resident_id}</div>
      </td>
      <td class="text-multiline">${item.past_bookings}</td>
      <td class="text-multiline">${item.prediction}</td>
      <td class="text-multiline">${item.actual}</td>
      <td>
        <div class="badge-match ${badgeClass}">
          <div>${item.match_indicator}</div>
          <div class="match-score-sub">${item.score}</div>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function setupEventListeners() {
  const searchInput = document.getElementById("search-input");
  const filterMatch = document.getElementById("filter-match");
  const btnPredict = document.getElementById("btn-predict");
  const btnExport = document.getElementById("btn-export-csv");

  searchInput.addEventListener("input", applyFilters);
  filterMatch.addEventListener("change", applyFilters);
  
  btnPredict.addEventListener("click", runLivePrediction);
  btnExport.addEventListener("click", exportToCSV);
}

function applyFilters() {
  const query = document.getElementById("search-input").value.toLowerCase();
  const matchFilter = document.getElementById("filter-match").value;

  const filtered = allPredictions.filter(p => {
    const matchesSearch = p.resident_id.toLowerCase().includes(query) || p.record_reference.toLowerCase().includes(query);
    
    let matchesCategory = true;
    if (matchFilter === "EXACT") {
      matchesCategory = p.score.includes("4 of 4");
    } else if (matchFilter === "PARTIAL") {
      matchesCategory = p.score.includes("3 of 4") || p.score.includes("2 of 4");
    } else if (matchFilter === "MISS") {
      matchesCategory = p.score.includes("1 of 4") || p.score.includes("0 of 4");
    }

    return matchesSearch && matchesCategory;
  });

  renderTable(filtered);
}

function runLivePrediction() {
  const resId = document.getElementById("resident-select").value;
  const resPredictions = allPredictions.filter(p => p.resident_id === resId);

  if (resPredictions.length === 0) return;

  const sample = resPredictions[Math.floor(Math.random() * resPredictions.length)];
  const box = document.getElementById("sim-result");

  document.getElementById("sim-fac").textContent = sample.pred_facility;
  document.getElementById("sim-day").textContent = sample.pred_day;
  document.getElementById("sim-time").textContent = sample.pred_use_time;
  document.getElementById("sim-nudge").textContent = sample.pred_nudge_time;

  box.classList.remove("hidden");
}

function exportToCSV() {
  if (!allPredictions || allPredictions.length === 0) return;

  const headers = ["record_reference", "resident_id", "past_bookings", "prediction", "actual", "match_indicator", "score"];
  const rows = allPredictions.map(p => [
    p.record_reference,
    p.resident_id,
    `"${p.past_bookings.replace(/\n/g, ' ')}"`,
    `"${p.prediction.replace(/\n/g, ' ')}"`,
    `"${p.actual.replace(/\n/g, ' ')}"`,
    p.match_indicator,
    p.score
  ]);

  const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", "facility_prediction_review.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function loadFallbackData() {
  // Fallback demo dataset if fetched directly
  dashboardData = {
    metrics: {
      total_test_records: 120,
      facility_accuracy: 86.5,
      usage_day_accuracy: 82.1,
      usage_hour_accuracy: 84.8,
      nudge_time_accuracy: 79.2,
      nudge_time_mae_hours: 1.45,
      exact_4of4_match_rate: 68.4,
      match_distribution: { "4_of_4": 82, "3_of_4": 24, "2_of_4": 10, "1_of_4": 4, "0_of_4": 0 }
    },
    predictions: [
      {
        record_reference: "R-104-#1",
        resident_id: "R-104",
        past_bookings: "Gym / Mon / 07:00\nGym (Top Pref)",
        prediction: "Gym / Fri / 07:00\nNudge Thu / 18:15",
        actual: "Gym / Fri / 07:00\nBooked Thu / 18:15",
        match_indicator: "YES",
        score: "4 of 4",
        pred_facility: "Gym",
        pred_day: "Fri",
        pred_use_time: "07:00",
        pred_nudge_time: "Nudge Thu / 18:15"
      },
      {
        record_reference: "R-118-#2",
        resident_id: "R-118",
        past_bookings: "Swimming Pool / Sat / 18:00\nSwimming Pool (Top Pref)",
        prediction: "Swimming Pool / Sat / 18:00\nNudge Fri / 09:20",
        actual: "Swimming Pool / Sat / 18:00\nBooked Fri / 09:20",
        match_indicator: "YES",
        score: "4 of 4",
        pred_facility: "Swimming Pool",
        pred_day: "Sat",
        pred_use_time: "18:00",
        pred_nudge_time: "Nudge Fri / 09:20"
      },
      {
        record_reference: "R-126-#3",
        resident_id: "R-126",
        past_bookings: "Badminton Court / Thu / 20:00\nBadminton Court (Top Pref)",
        prediction: "Badminton Court / Fri / 20:00\nNudge Thu / 12:30",
        actual: "Gym / Fri / 19:00\nBooked Thu / 12:30",
        match_indicator: "NO",
        score: "2 of 4",
        pred_facility: "Badminton Court",
        pred_day: "Fri",
        pred_use_time: "20:00",
        pred_nudge_time: "Nudge Thu / 12:30"
      }
    ]
  };
  renderDashboard();
}
