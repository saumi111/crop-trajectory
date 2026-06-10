import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import numpy as np

# Load both seasons
with open("sar_timeseries_test.json") as f:
    season_2425 = [d for d in json.load(f) if d.get("available")]

with open("sar_2023_24.json") as f:
    season_2324 = [d for d in json.load(f) if d.get("available")]

# Convert to day-of-year for comparison
def to_doy(date_str):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    # Normalise to season day (Oct 1 = day 1)
    oct1 = datetime.datetime(d.year if d.month >= 10 else d.year-1, 10, 1)
    return (d - oct1).days

# Build indexed series
doy_2324 = [(to_doy(d["date"]), d["rvi"]) for d in season_2324]
doy_2425 = [(to_doy(d["date"]), d["rvi"]) for d in season_2425]

# Sort by day
doy_2324.sort(key=lambda x: x[0])
doy_2425.sort(key=lambda x: x[0])

x_2324 = [d[0] for d in doy_2324]
y_2324 = [d[1] for d in doy_2324]
x_2425 = [d[0] for d in doy_2425]
y_2425 = [d[1] for d in doy_2425]

# Calculate deviation at matching points
deviations = []
for i, (day, rvi_curr) in enumerate(doy_2425):
    # Find closest point in previous season
    closest = min(doy_2324, key=lambda x: abs(x[0]-day))
    if abs(closest[0] - day) <= 8:  # Within 8 days
        dev_pct = (rvi_curr - closest[1]) / closest[1] * 100
        deviations.append((day, dev_pct, rvi_curr, closest[1]))

# Wheat growth stage labels (days from Oct 1)
stages = {
    "Sowing": 0,
    "Emergence": 24,
    "Tillering": 61,
    "Stem ext.": 166,
    "Heading": 227,
    "Grain fill": 257,
    "Harvest": 293
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
fig.patch.set_facecolor('#0a1525')
ax1.set_facecolor('#0a1525')
ax2.set_facecolor('#0a1525')

# Top panel - both seasons
ax1.plot(x_2324, y_2324, color='#aaaaaa', linewidth=2,
         marker='o', markersize=3, label='2023/24 Season (baseline)',
         linestyle='--', alpha=0.8)
ax1.plot(x_2425, y_2425, color='#4da6ff', linewidth=2.5,
         marker='o', markersize=4, label='2024/25 Season (current)')
ax1.fill_between(x_2425, y_2425, alpha=0.15, color='#4da6ff')

# Shade deviation areas
for day, dev_pct, curr, base in deviations:
    color = '#ff4444' if dev_pct < -5 else '#44ff44' if dev_pct > 5 else '#ffaa00'
    ax1.axvline(x=day, color=color, alpha=0.15, linewidth=8)

# Stage lines
for stage, day in stages.items():
    ax1.axvline(x=day, color='#ffffff', alpha=0.2, linestyle=':', linewidth=1)
    ax1.text(day+1, 0.74, stage, color='#ffffff', fontsize=7,
             rotation=45, ha='left', alpha=0.6)

ax1.set_ylabel('SAR RVI', color='white', fontsize=11)
ax1.set_title('Lincolnshire Winter Wheat — Development Trajectory Comparison\n'
              'SAR Sentinel-1 · Field: 53.23°N 0.54°W · Cube Earth',
              color='white', fontsize=12, fontweight='bold', pad=15)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#333')
ax1.set_ylim(0.45, 0.78)
ax1.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=10)
ax1.grid(alpha=0.1, color='white')

# Bottom panel - deviation
dev_days = [d[0] for d in deviations]
dev_vals = [d[1] for d in deviations]
dev_colors = ['#ff4444' if v < -5 else '#44ff44' if v > 5 else '#ffaa00'
              for v in dev_vals]

bars = ax2.bar(dev_days, dev_vals, color=dev_colors, alpha=0.8, width=10)
ax2.axhline(y=0, color='white', linewidth=1, alpha=0.5)
ax2.axhline(y=10, color='#44ff44', linewidth=0.5, alpha=0.3, linestyle='--')
ax2.axhline(y=-10, color='#ff4444', linewidth=0.5, alpha=0.3, linestyle='--')

# Flag significant deviations
for day, dev in zip(dev_days, dev_vals):
    if abs(dev) > 10:
        ax2.annotate(f'{dev:.0f}%',
                    xy=(day, dev),
                    xytext=(day, dev + (3 if dev > 0 else -3)),
                    color='white', fontsize=8, ha='center',
                    fontweight='bold')

ax2.set_ylabel('Deviation from\n2023/24 baseline (%)', color='white', fontsize=10)
ax2.set_xlabel('Days from 1 October (season start)', color='white', fontsize=11)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#333')
ax2.grid(alpha=0.1, color='white')

green_p = mpatches.Patch(color='#44ff44', alpha=0.8, label='Above baseline (>5%)')
amber_p = mpatches.Patch(color='#ffaa00', alpha=0.8, label='Near baseline (±5%)')
red_p = mpatches.Patch(color='#ff4444', alpha=0.8, label='Below baseline (<-5%)')
ax2.legend(handles=[green_p, amber_p, red_p],
           facecolor='#1a2a3a', labelcolor='white', fontsize=9)

plt.tight_layout(pad=2)
plt.savefig('wheat_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0a1525')
print("Saved: wheat_comparison.png")
plt.close()

# Print summary
print("\n=== Deviation Summary ===")
significant = [(d,v) for d,v in zip(dev_days, dev_vals) if abs(v) > 5]
for day, dev in significant:
    status = "ABOVE" if dev > 0 else "BELOW"
    print(f"Day {day}: {dev:.1f}% {status} baseline")
