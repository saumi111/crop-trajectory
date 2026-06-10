import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime

# Load full SAR time series
with open("sar_timeseries_test.json") as f:
    data = json.load(f)

dates = [d["date"] for d in data if d.get("available")]
rvi = [d["rvi"] for d in data if d.get("available")]
vv = [d["vv"] for d in data if d.get("available")]
vh = [d["vh"] for d in data if d.get("available")]

# Convert dates
x = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]

# Calculate velocity
velocity = [None]
for i in range(1, len(rvi)):
    days = (x[i] - x[i-1]).days
    vel = (rvi[i] - rvi[i-1]) / days if days > 0 else 0
    velocity.append(vel)

# Wheat growth stages (approximate dates for UK winter wheat)
stages = {
    "Sowing": "2024-10-01",
    "Emergence": "2024-10-25",
    "Tillering": "2024-12-01",
    "Stem ext.": "2025-03-15",
    "Heading": "2025-05-15",
    "Grain fill": "2025-06-15",
    "Harvest": "2025-07-20"
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.patch.set_facecolor('#0a1525')
ax1.set_facecolor('#0a1525')
ax2.set_facecolor('#0a1525')

# Plot RVI
ax1.plot(x, rvi, color='#4da6ff', linewidth=2, marker='o', markersize=4, label='RVI 2024-25')
ax1.fill_between(x, rvi, alpha=0.2, color='#4da6ff')

# Add stage lines
for stage, date_str in stages.items():
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    ax1.axvline(x=d, color='#ffffff', alpha=0.3, linestyle='--', linewidth=1)
    ax1.text(d, max(rvi)*1.02, stage, color='#ffffff', fontsize=7, 
             rotation=45, ha='left', alpha=0.7)

ax1.set_ylabel('SAR RVI', color='white', fontsize=11)
ax1.set_title('Lincolnshire Winter Wheat — SAR Development Trajectory 2024/25', 
              color='white', fontsize=13, fontweight='bold', pad=15)
ax1.tick_params(colors='white')
ax1.spines['bottom'].set_color('#333')
ax1.spines['top'].set_color('#333')
ax1.spines['left'].set_color('#333')
ax1.spines['right'].set_color('#333')
ax1.set_ylim(0.45, 0.75)
ax1.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=9)
ax1.grid(alpha=0.1, color='white')

# Plot velocity
vel_colors = ['#ff4444' if v and v < -0.002 else '#44ff44' if v and v > 0.001 else '#ffaa00' 
              for v in velocity]
vel_values = [v if v else 0 for v in velocity]

ax2.bar(x, vel_values, color=vel_colors, alpha=0.7, width=10)
ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
ax2.set_ylabel('Development Velocity\n(RVI change/day)', color='white', fontsize=10)
ax2.set_xlabel('Date', color='white', fontsize=11)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#333')
ax2.spines['top'].set_color('#333')
ax2.spines['left'].set_color('#333')
ax2.spines['right'].set_color('#333')
ax2.grid(alpha=0.1, color='white')

# Legend for velocity
green = mpatches.Patch(color='#44ff44', alpha=0.7, label='Developing')
orange = mpatches.Patch(color='#ffaa00', alpha=0.7, label='Stable')
red = mpatches.Patch(color='#ff4444', alpha=0.7, label='Declining')
ax2.legend(handles=[green, orange, red], facecolor='#1a2a3a', 
           labelcolor='white', fontsize=9)

plt.tight_layout(pad=2)
plt.savefig('wheat_trajectory_2024_25.png', dpi=150, bbox_inches='tight',
            facecolor='#0a1525')
print("Chart saved: wheat_trajectory_2024_25.png")
plt.close()
