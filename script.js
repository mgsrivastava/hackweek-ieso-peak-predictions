// Function to get day of year (1-365)
function getDayOfYear(date) {
  const start = new Date(date.getFullYear(), 0, 0);
  const diff = date - start;
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

// Function to parse CSV data for demand
function parseCSVData(csvText) {
  const lines = csvText.trim().split("\n");

  // Find the header row
  let startIndex = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes("Date") && lines[i].includes("Hour")) {
      startIndex = i;
      break;
    }
  }

  // Process only the data rows
  return lines
    .slice(startIndex + 1)
    .map((line) => {
      const values = line.split(",").map((val) => val.trim());
      const date = values[0];
      const hour = values[1];
      const ontarioDemand = values[3]; // Ontario Demand is the fourth column
      const timestamp = new Date(`${date}T${hour.padStart(2, "0")}:00:00`);
      return {
        date: date,
        hour: parseInt(hour),
        dayOfYear: getDayOfYear(timestamp) + parseInt(hour) / 24,
        demand1: parseFloat(ontarioDemand), // Using Ontario Demand instead of Market Demand
        timestamp: timestamp,
      };
    })
    .filter((item) => !isNaN(item.demand1) && !isNaN(item.timestamp.getTime()));
}

// Function to parse weather CSV data
function parseWeatherData(csvText) {
  const lines = csvText.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""));
  const maxTempIndex = headers.findIndex((h) => h.includes("Max Temp"));
  const minTempIndex = headers.findIndex((h) => h.includes("Min Temp"));
  const dateIndex = headers.findIndex((h) => h.includes("Date/Time"));

  return lines
    .slice(1)
    .map((line) => {
      const values = line.split(",").map((v) => v.trim().replace(/"/g, ""));
      const dateStr = values[dateIndex];
      const maxTemp =
        values[maxTempIndex] === "M" ? null : parseFloat(values[maxTempIndex]);
      const minTemp =
        values[minTempIndex] === "M" ? null : parseFloat(values[minTempIndex]);

      if (!dateStr || dateStr === "") return null;

      const date = new Date(dateStr + "T00:00:00");

      return {
        date,
        maxTemp,
        minTemp,
      };
    })
    .filter(
      (d) =>
        d !== null &&
        d.maxTemp !== null &&
        d.minTemp !== null &&
        !isNaN(d.date.getTime())
    );
}

// Function to create the demand chart
function createDemandChart(data2021, data2022, data2023) {
  const ctx = document.getElementById("demandChart").getContext("2d");

  new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "2021 Demand",
          data: data2021.map((d) => ({ x: d.dayOfYear, y: d.demand1 })),
          borderColor: "rgb(255, 99, 132)",
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: "2022 Demand",
          data: data2022.map((d) => ({ x: d.dayOfYear, y: d.demand1 })),
          borderColor: "rgb(54, 162, 235)",
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: "2023 Demand",
          data: data2023.map((d) => ({ x: d.dayOfYear, y: d.demand1 })),
          borderColor: "rgb(75, 192, 192)",
          borderWidth: 1,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: "Yearly Demand Comparison (Overlaid)",
        },
        tooltip: {
          callbacks: {
            title: function (context) {
              const dayOfYear = context[0].parsed.x;
              const day = Math.floor(dayOfYear);
              const hour = Math.round((dayOfYear - day) * 24);
              return `Day ${day}, Hour ${hour}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          min: 1,
          max: 365,
          title: {
            display: true,
            text: "Day of Year",
          },
          ticks: {
            stepSize: 30,
            callback: function (value) {
              return `Day ${value}`;
            },
          },
        },
        y: {
          title: {
            display: true,
            text: "Demand",
          },
        },
      },
    },
  });
}

// Function to create the weather chart
function createWeatherChart(weatherData) {
  const ctx = document.getElementById("weatherChart").getContext("2d");

  console.log("Weather data points:", weatherData.length);

  new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Maximum Temperature (°C)",
          data: weatherData.map((d) => ({ x: d.date.getTime(), y: d.maxTemp })),
          borderColor: "rgb(255, 99, 132)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Minimum Temperature (°C)",
          data: weatherData.map((d) => ({ x: d.date.getTime(), y: d.minTemp })),
          borderColor: "rgb(54, 162, 235)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: "2024 Temperature Data",
        },
      },
      scales: {
        x: {
          type: "time",
          time: {
            unit: "month",
            displayFormats: {
              month: "MMM yyyy",
            },
          },
          title: {
            display: true,
            text: "Date",
          },
          min: new Date("2024-01-01").getTime(),
          max: new Date("2024-12-31").getTime(),
        },
        y: {
          title: {
            display: true,
            text: "Temperature (°C)",
          },
          min: -15,
          max: 35,
        },
      },
    },
  });
}

// Add this function to create the prediction chart
function createPredictionChart(predictions) {
  const ctx = document.getElementById("predictionChart").getContext("2d");

  new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Random Forest Predictions",
          data: predictions.dates.map((day, index) => ({
            x: day,
            y: predictions.rf_predictions[index],
          })),
          borderColor: "rgb(153, 102, 255)",
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: "Random Forest Predictions for 2024",
        },
        legend: {
          display: true,
          position: "top",
        },
      },
      scales: {
        x: {
          type: "linear",
          min: 1,
          max: 365,
          title: {
            display: true,
            text: "Day of Year",
          },
          ticks: {
            stepSize: 30,
            callback: function (value) {
              return `Day ${value}`;
            },
          },
        },
        y: {
          title: {
            display: true,
            text: "Predicted Demand (MW)",
          },
        },
      },
    },
  });
}

// Add this function to populate peak demand tables
function populatePeakTable(data, year) {
  // Sort data by demand in descending order
  const sortedData = [...data].sort((a, b) => b.demand1 - a.demand1);

  // Get top 10 peaks
  const top10 = sortedData.slice(0, 10);

  // Get the table body
  const tableBody = document.querySelector(`#peaks${year} tbody`);

  // Clear existing rows
  tableBody.innerHTML = "";

  // Add rows for each peak
  top10.forEach((peak, index) => {
    const row = document.createElement("tr");

    // Format the date nicely
    const formattedDate = peak.timestamp.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });

    // Format the hour with leading zero and proper suffix
    const formattedHour = peak.hour.toString().padStart(2, "0") + ":00";

    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${formattedDate}</td>
      <td>${formattedHour}</td>
      <td>${Math.round(peak.demand1).toLocaleString()} MW</td>
    `;

    tableBody.appendChild(row);
  });
}

// Function to populate the predictions table
function populatePredictionTable(predictions) {
  // Create array of day-prediction pairs with day of week info
  const predictionPairs = predictions.dates.map((day, index) => {
    const date = new Date(2024, 0); // January 1, 2024
    date.setDate(day);
    const dayOfWeek = date.toLocaleDateString("en-US", { weekday: "long" });
    const isWeekend = date.getDay() === 0 || date.getDay() === 6; // 0 is Sunday, 6 is Saturday

    return {
      day: day,
      demand: predictions.rf_predictions[index], // Use Random Forest predictions
      dayOfWeek: dayOfWeek,
      isWeekend: isWeekend,
      date: date,
    };
  });

  // Remove duplicates and weekends
  const uniquePredictions = predictionPairs.reduce((acc, curr) => {
    if (!curr.isWeekend && !acc.some((item) => item.day === curr.day)) {
      acc.push(curr);
    }
    return acc;
  }, []);

  // Sort by predicted demand in descending order
  const sortedPredictions = uniquePredictions.sort(
    (a, b) => b.demand - a.demand
  );

  // Get top 20 peaks (changed from 10)
  const top20 = sortedPredictions.slice(0, 20);

  // Get the table body
  const tableBody = document.querySelector("#peaks2024 tbody");

  // Clear existing rows
  tableBody.innerHTML = "";

  // Add rows for each peak
  top20.forEach((peak, index) => {
    const row = document.createElement("tr");

    const formattedDate = peak.date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });

    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${formattedDate} (Day ${peak.day})</td>
      <td>${peak.dayOfWeek}</td>
      <td>${Math.round(peak.demand).toLocaleString()} MW</td>
    `;

    tableBody.appendChild(row);
  });
}

// Update the Promise.all section to include loading predictions
Promise.all([
  fetch("past_data/PUB_Demand_2021.csv").then((response) => response.text()),
  fetch("past_data/PUB_Demand_2022.csv").then((response) => response.text()),
  fetch("past_data/PUB_Demand_2023.csv").then((response) => response.text()),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_01.csv").then(
    (response) => response.text()
  ),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_02.csv").then(
    (response) => response.text()
  ),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_03.csv").then(
    (response) => response.text()
  ),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_04.csv").then(
    (response) => response.text()
  ),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_05.csv").then(
    (response) => response.text()
  ),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_06.csv").then(
    (response) => response.text()
  ),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_07.csv").then(
    (response) => response.text()
  ),
  fetch("2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_08.csv").then(
    (response) => response.text()
  ),
  fetch("predictions.json").then((response) => {
    console.log("Predictions response status:", response.status);
    return response.json();
  }),
])
  .then(([data2021, data2022, data2023, ...rest]) => {
    const weatherFiles = rest.slice(0, -1);
    const predictions = rest[rest.length - 1];

    console.log("Loaded predictions:", predictions);

    const parsed2021 = parseCSVData(data2021);
    const parsed2022 = parseCSVData(data2022);
    const parsed2023 = parseCSVData(data2023);

    const weatherData = weatherFiles
      .map((file) => parseWeatherData(file))
      .flat()
      .sort((a, b) => a.date - b.date);

    console.log("Creating charts...");
    // Create charts
    createDemandChart(parsed2021, parsed2022, parsed2023);
    createWeatherChart(weatherData);
    console.log("Creating prediction chart with data:", predictions);
    createPredictionChart(predictions);

    console.log("Populating tables...");
    // Populate peak tables
    populatePeakTable(parsed2021, 2021);
    populatePeakTable(parsed2022, 2022);
    populatePeakTable(parsed2023, 2023);
    populatePredictionTable(predictions);
  })
  .catch((error) => {
    console.error("Error loading data:", error);
    console.error("Stack trace:", error.stack);
  });
