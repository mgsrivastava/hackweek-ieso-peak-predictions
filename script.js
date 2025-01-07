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

  // Process data rows and group by date
  const dailyData = new Map();

  lines.slice(startIndex + 1).forEach((line) => {
    const values = line.split(",").map((val) => val.trim());
    const date = values[0];
    const demand = parseFloat(values[3]); // Ontario Demand is the fourth column

    if (!isNaN(demand)) {
      if (!dailyData.has(date)) {
        dailyData.set(date, []);
      }
      dailyData.get(date).push(demand);
    }
  });

  // Calculate daily averages
  return Array.from(dailyData.entries())
    .map(([date, demands]) => {
      const timestamp = new Date(`${date}T00:00:00`);
      return {
        date: date,
        dayOfYear: getDayOfYear(timestamp),
        demand1: demands.reduce((sum, val) => sum + val, 0) / demands.length, // Average demand
        timestamp: timestamp,
      };
    })
    .filter((item) => !isNaN(item.demand1) && !isNaN(item.timestamp.getTime()))
    .sort((a, b) => a.timestamp - b.timestamp);
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
function createDemandChart(
  data2018,
  data2019,
  data2020,
  data2021,
  data2022,
  data2023
) {
  const ctx = document.getElementById("demandChart").getContext("2d");

  // Combine all data into a single array and sort chronologically
  const allData = [
    ...data2018,
    ...data2019,
    ...data2020,
    ...data2021,
    ...data2022,
    ...data2023,
  ].sort((a, b) => a.timestamp - b.timestamp);

  new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Historical Average Daily Demand",
          data: allData.map((d) => ({
            x: d.timestamp.getTime(),
            y: d.demand1,
          })),
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
          text: "Historical Average Daily Demand (2018-2023)",
        },
        tooltip: {
          callbacks: {
            title: function (context) {
              const date = new Date(context[0].parsed.x);
              return date.toLocaleDateString("en-US", {
                year: "numeric",
                month: "short",
                day: "numeric",
              });
            },
          },
        },
        legend: {
          position: "right",
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
          min: new Date("2018-01-01").getTime(),
          max: new Date("2023-12-31").getTime(),
        },
        y: {
          title: {
            display: true,
            text: "Average Daily Demand (MW)",
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

    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${formattedDate}</td>
      <td>Daily Average</td>
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

// Function to create the historical weather chart
function createHistoricalWeatherChart(
  weatherData2018,
  weatherData2019,
  weatherData2020,
  weatherData2021,
  weatherData2022,
  weatherData2023
) {
  const ctx = document
    .getElementById("historicalWeatherChart")
    .getContext("2d");

  // Combine all weather data
  const allWeatherData = [
    ...weatherData2018,
    ...weatherData2019,
    ...weatherData2020,
    ...weatherData2021,
    ...weatherData2022,
    ...weatherData2023,
  ].sort((a, b) => a.date - b.date);

  new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Maximum Temperature (°C)",
          data: allWeatherData.map((d) => ({
            x: d.date.getTime(),
            y: d.maxTemp,
          })),
          borderColor: "rgb(255, 99, 132)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Minimum Temperature (°C)",
          data: allWeatherData.map((d) => ({
            x: d.date.getTime(),
            y: d.minTemp,
          })),
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
          text: "Historical Temperature Data (2018-2023)",
        },
        legend: {
          position: "right",
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
          min: new Date("2018-01-01").getTime(),
          max: new Date("2023-12-31").getTime(),
        },
        y: {
          title: {
            display: true,
            text: "Temperature (°C)",
          },
          min: -30,
          max: 35,
        },
      },
    },
  });
}

// Update the Promise.all section
Promise.all([
  // Demand data fetches
  fetch("past_data/PUB_Demand_2018.csv").then((response) => response.text()),
  fetch("past_data/PUB_Demand_2019.csv").then((response) => response.text()),
  fetch("past_data/PUB_Demand_2020.csv").then((response) => response.text()),
  fetch("past_data/PUB_Demand_2021.csv").then((response) => response.text()),
  fetch("past_data/PUB_Demand_2022.csv").then((response) => response.text()),
  fetch("past_data/PUB_Demand_2023.csv").then((response) => response.text()),
  // Historical weather data fetches
  Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      fetch(
        `2018_weather_data/en_climate_daily_ON_6158359_2018_P1D_${String(
          i + 1
        ).padStart(2, "0")}.csv`
      )
        .then((response) => response.text())
        .catch(() => null)
    )
  ),
  Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      fetch(
        `2019_weather_data/en_climate_daily_ON_6158359_2019_P1D_${String(
          i + 1
        ).padStart(2, "0")}.csv`
      )
        .then((response) => response.text())
        .catch(() => null)
    )
  ),
  Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      fetch(
        `2020_weather_data/en_climate_daily_ON_6158359_2020_P1D_${String(
          i + 1
        ).padStart(2, "0")}.csv`
      )
        .then((response) => response.text())
        .catch(() => null)
    )
  ),
  Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      fetch(
        `2021_weather_data/en_climate_daily_ON_6158359_2021_P1D_${String(
          i + 1
        ).padStart(2, "0")}.csv`
      )
        .then((response) => response.text())
        .catch(() => null)
    )
  ),
  Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      fetch(
        `2022_weather_data/en_climate_daily_ON_6158359_2022_P1D_${String(
          i + 1
        ).padStart(2, "0")}.csv`
      )
        .then((response) => response.text())
        .catch(() => null)
    )
  ),
  Promise.all(
    Array.from({ length: 12 }, (_, i) =>
      fetch(
        `2023_weather_data/en_climate_daily_ON_6158359_2023_P1D_${String(
          i + 1
        ).padStart(2, "0")}.csv`
      )
        .then((response) => response.text())
        .catch(() => null)
    )
  ),
  // 2024 weather data fetches
  Promise.all(
    Array.from({ length: 8 }, (_, i) =>
      fetch(
        `2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_${String(
          i + 1
        ).padStart(2, "0")}.csv`
      ).then((response) => response.text())
    )
  ),
  // Predictions fetch
  fetch("predictions.json").then((response) => {
    console.log("Predictions response status:", response.status);
    return response.json();
  }),
]).then(
  ([
    data2018,
    data2019,
    data2020,
    data2021,
    data2022,
    data2023,
    weather2018Files,
    weather2019Files,
    weather2020Files,
    weather2021Files,
    weather2022Files,
    weather2023Files,
    weather2024Files,
    predictions,
  ]) => {
    console.log("Loaded predictions:", predictions);

    // Parse demand data
    const parsed2018 = parseCSVData(data2018);
    const parsed2019 = parseCSVData(data2019);
    const parsed2020 = parseCSVData(data2020);
    const parsed2021 = parseCSVData(data2021);
    const parsed2022 = parseCSVData(data2022);
    const parsed2023 = parseCSVData(data2023);

    // Parse historical weather data
    const weather2018 = weather2018Files
      .filter((f) => f !== null)
      .map((file) => parseWeatherData(file))
      .flat();
    const weather2019 = weather2019Files
      .filter((f) => f !== null)
      .map((file) => parseWeatherData(file))
      .flat();
    const weather2020 = weather2020Files
      .filter((f) => f !== null)
      .map((file) => parseWeatherData(file))
      .flat();
    const weather2021 = weather2021Files
      .filter((f) => f !== null)
      .map((file) => parseWeatherData(file))
      .flat();
    const weather2022 = weather2022Files
      .filter((f) => f !== null)
      .map((file) => parseWeatherData(file))
      .flat();
    const weather2023 = weather2023Files
      .filter((f) => f !== null)
      .map((file) => parseWeatherData(file))
      .flat();

    // Parse 2024 weather data
    const weatherData2024 = weather2024Files
      .map((file) => parseWeatherData(file))
      .flat()
      .sort((a, b) => a.date - b.date);

    console.log("Creating charts...");
    // Create charts
    createDemandChart(
      parsed2018,
      parsed2019,
      parsed2020,
      parsed2021,
      parsed2022,
      parsed2023
    );
    createHistoricalWeatherChart(
      weather2018,
      weather2019,
      weather2020,
      weather2021,
      weather2022,
      weather2023
    );
    createWeatherChart(weatherData2024);
    console.log("Creating prediction chart with data:", predictions);
    createPredictionChart(predictions);

    console.log("Populating tables...");
    // Populate peak tables
    populatePeakTable(parsed2018, 2018);
    populatePeakTable(parsed2019, 2019);
    populatePeakTable(parsed2020, 2020);
    populatePeakTable(parsed2021, 2021);
    populatePeakTable(parsed2022, 2022);
    populatePeakTable(parsed2023, 2023);
    populatePredictionTable(predictions);
  }
);
