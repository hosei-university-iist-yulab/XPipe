#!/usr/bin/env python3
"""
Create Smart Grid/IoT QA Datasets for XPipe Experiments
========================================================

Generates 2 domain-specific QA datasets:
1. EV/Battery Management (ev_battery)
2. Solar PV Systems (solar_pv)

Based on public domain knowledge from:
- EV: ACN-Data (Caltech), NREL standards, IEEE charging protocols
- Solar: Open PV Project, PVDAQ, NREL solar resources

Output: datasets/{ev_battery,solar_pv}/{cases.jsonl, queries.jsonl}
"""

import json
from pathlib import Path

# ===========================================================================
# EV/Battery Management Dataset
# ===========================================================================

EV_BATTERY_DOCS = [
    {
        "id": "ev_1",
        "text": "Electric Vehicle Battery Basics:\n\nBattery Capacity: Measured in kilowatt-hours (kWh). Common ranges:\n- Compact EVs: 30-40 kWh (150-200 miles range)\n- Mid-size EVs: 60-80 kWh (250-350 miles range)\n- Luxury EVs: 100+ kWh (400+ miles range)\n\nBattery Degradation: Typically 2-3% per year. Factors affecting degradation:\n- Fast charging frequency (>50 kW DC charging)\n- Extreme temperatures (below 0°C or above 40°C)\n- Deep discharge cycles (below 10% state of charge)\n- High state of charge storage (>90% for extended periods)\n\nBattery Management System (BMS): Monitors cell voltages, temperature, and state of charge to ensure safe operation and maximize battery life."
    },
    {
        "id": "ev_2",
        "text": "EV Charging Levels and Standards:\n\n**Level 1 (120V AC)**:\n- Power: 1.4-1.9 kW\n- Charging time: 40-50 hours for full charge (60 kWh battery)\n- Use case: Home charging, overnight parking\n\n**Level 2 (240V AC)**:\n- Power: 3.3-19.2 kW (typically 6.6-7.2 kW)\n- Charging time: 4-8 hours for full charge\n- Use case: Home, workplace, public charging stations\n- Standards: SAE J1772 (North America), IEC 62196 Type 2 (Europe)\n\n**Level 3 (DC Fast Charging)**:\n- Power: 50-350 kW\n- Charging time: 20-60 minutes to 80% charge\n- Use case: Highway corridors, rapid charging stations\n- Standards: CHAdeMO, CCS (Combined Charging System), Tesla Supercharger\n\n**Note**: Fast charging should be limited to 20% of charging sessions to minimize battery degradation."
    },
    {
        "id": "ev_3",
        "text": "Battery Thermal Management:\n\nOptimal Operating Temperature: 20-35°C (68-95°F)\n\n**Cold Weather Challenges**:\n- Reduced battery capacity (up to 40% loss at -20°C)\n- Slower charging rates (50% slower below 0°C)\n- Increased internal resistance\n\n**Thermal Management Solutions**:\n1. Active Liquid Cooling: Circulates coolant through battery pack\n2. Resistive Heating: Pre-heats battery before charging\n3. Heat Pump Systems: Maintains optimal temperature range\n4. Battery Pre-conditioning: Remote activation before driving\n\n**Hot Weather Challenges**:\n- Accelerated degradation (doubles every 10°C above optimal)\n- Thermal runaway risk above 60°C\n- Reduced charging power to prevent overheating\n\n**Best Practices**: Park in shade, use climate pre-conditioning while plugged in, avoid fast charging in extreme heat."
    },
    {
        "id": "ev_4",
        "text": "Smart Charging and V2G (Vehicle-to-Grid):\n\n**Smart Charging Benefits**:\n- Load balancing: Shift charging to off-peak hours (typically 11pm-6am)\n- Cost savings: 30-50% reduction using time-of-use rates\n- Grid stabilization: Reduce peak demand stress\n\n**V2G Technology**:\n- Bidirectional charging: EV battery can supply power back to grid\n- Grid services: Frequency regulation, demand response, peak shaving\n- Revenue potential: $500-1,500/year per vehicle (varies by market)\n- Requirements: CHAdeMO or CCS bidirectional charger, utility agreement\n\n**ISO 15118 Protocol**: Enables plug-and-charge authentication, dynamic pricing, and automated grid services\n\n**Challenges**: Battery degradation from extra cycles, regulatory approval, infrastructure costs"
    },
    {
        "id": "ev_5",
        "text": "Battery State Estimation:\n\n**State of Charge (SOC)**:\n- Definition: Remaining battery capacity as percentage\n- Estimation methods:\n  1. Coulomb counting: Integrates current over time (±5% accuracy)\n  2. Voltage-based: Open-circuit voltage correlation (±10% accuracy)\n  3. Kalman filtering: Combines multiple measurements (±2% accuracy)\n  4. Machine learning: Neural networks trained on historical data (±1% accuracy)\n\n**State of Health (SOH)**:\n- Definition: Current capacity vs original capacity\n- Typical values: 100% (new) → 80% (warranty threshold) → 70% (degraded)\n- Measurement: Full charge-discharge cycle, impedance spectroscopy\n\n**State of Power (SOP)**:\n- Definition: Maximum power available for acceleration/regeneration\n- Factors: Temperature, SOC, cell impedance\n- Critical for performance management and safety limits\n\n**Importance**: Accurate estimation prevents over-discharge, optimizes charging, and extends battery life by 15-25%."
    }
]

EV_BATTERY_QUERIES = [
    {"id": "evq1", "text": "What is the typical battery capacity range for mid-size electric vehicles?", "ref": "Mid-size EVs typically have battery capacities of 60-80 kWh, providing a range of 250-350 miles."},
    {"id": "evq2", "text": "How much battery degradation should I expect per year?", "ref": "Electric vehicle batteries typically degrade 2-3% per year, affected by fast charging frequency, extreme temperatures, and charging habits."},
    {"id": "evq3", "text": "What is the charging time for Level 2 charging?", "ref": "Level 2 charging (240V AC) typically provides 6.6-7.2 kW power and takes 4-8 hours for a full charge."},
    {"id": "evq4", "text": "What are the main EV DC fast charging standards?", "ref": "The main DC fast charging standards are CHAdeMO, CCS (Combined Charging System), and Tesla Supercharger, providing 50-350 kW power."},
    {"id": "evq5", "text": "What is the optimal temperature range for EV battery operation?", "ref": "The optimal operating temperature for EV batteries is 20-35°C (68-95°F)."},
    {"id": "evq6", "text": "How does cold weather affect EV battery performance?", "ref": "Cold weather can reduce battery capacity by up to 40% at -20°C and slow charging rates by 50% below 0°C due to increased internal resistance."},
    {"id": "evq7", "text": "What is V2G technology and how much revenue can it generate?", "ref": "V2G (Vehicle-to-Grid) enables bidirectional charging where EVs supply power back to the grid, potentially generating $500-1,500 per year per vehicle for grid services."},
    {"id": "evq8", "text": "What are the benefits of smart charging?", "ref": "Smart charging enables load balancing by shifting to off-peak hours, provides 30-50% cost savings using time-of-use rates, and helps stabilize the grid."},
    {"id": "evq9", "text": "What is State of Health (SOH) for EV batteries?", "ref": "State of Health (SOH) is the ratio of current battery capacity to original capacity, typically declining from 100% (new) to 80% (warranty threshold)."},
    {"id": "evq10", "text": "What is the most accurate method for estimating battery State of Charge?", "ref": "Machine learning methods using neural networks trained on historical data achieve the highest accuracy (±1%) for State of Charge estimation."},
    {"id": "evq11", "text": "How often should I use DC fast charging?", "ref": "Fast charging should be limited to 20% of charging sessions to minimize battery degradation."},
    {"id": "evq12", "text": "What is the ISO 15118 protocol used for?", "ref": "ISO 15118 enables plug-and-charge authentication, dynamic pricing, and automated grid services for electric vehicles."},
    {"id": "evq13", "text": "How can I extend my EV battery life?", "ref": "Extend battery life by limiting fast charging, avoiding extreme temperatures, not storing at high charge (>90%), and using pre-conditioning while plugged in."},
    {"id": "evq14", "text": "What happens to battery capacity in hot weather?", "ref": "Hot weather accelerates battery degradation (doubles every 10°C above optimal) and can trigger thermal runaway above 60°C."}
]

# ===========================================================================
# Solar PV Systems Dataset
# ===========================================================================

SOLAR_PV_DOCS = [
    {
        "id": "pv_1",
        "text": "Photovoltaic System Components:\n\n**Solar Panels (PV Modules)**:\n- Types: Monocrystalline (18-22% efficiency), Polycrystalline (15-17%), Thin-film (10-12%)\n- Typical lifespan: 25-30 years with 0.5% annual degradation\n- Power rating: 300-450 W per panel (residential), 500-700 W (commercial)\n\n**Inverters**:\n- String inverters: Centralized, cost-effective ($0.10-0.20/W)\n- Microinverters: Per-panel optimization, higher reliability ($0.40-0.60/W)\n- Power optimizers: Hybrid approach, module-level monitoring ($0.20-0.30/W)\n- Efficiency: 95-99% conversion from DC to AC\n\n**Balance of System (BOS)**:\n- Mounting structures: Roof-mount, ground-mount, tracking systems\n- Wiring and combiner boxes\n- AC/DC disconnects and circuit protection\n- Monitoring systems and data loggers\n\n**Energy Storage (Optional)**:\n- Lithium-ion batteries: 10-15 kWh residential, 90-95% round-trip efficiency\n- Backup power capability, self-consumption optimization"
    },
    {
        "id": "pv_2",
        "text": "Solar Panel Performance Factors:\n\n**Temperature Effects**:\n- Standard Test Conditions (STC): 25°C, 1000 W/m² irradiance\n- Temperature coefficient: -0.3 to -0.5% per °C above 25°C\n- Example: 400W panel at 45°C produces ~368W (8% loss)\n\n**Irradiance Variations**:\n- Peak sun hours: Equivalent hours at 1000 W/m² (varies by location)\n- Typical ranges: 3-4 hours (northern US), 5-7 hours (southwest US)\n- Cloud cover impact: Reduces output to 10-25% of capacity\n\n**Shading Effects**:\n- Partial shading can disproportionately reduce output (50% shading = 80% loss)\n- Bypass diodes mitigate shading impact\n- Solution: Microinverters or power optimizers for shaded installations\n\n**Orientation and Tilt**:\n- Optimal azimuth: South-facing (northern hemisphere)\n- Optimal tilt: Equal to latitude (±15° acceptable)\n- Deviation impact: 10° from optimal = 1-3% production loss\n\n**Soiling and Degradation**:\n- Dust/dirt accumulation: 2-5% annual loss (higher in arid regions)\n- Snow coverage: Complete blockage until melted or removed\n- Annual degradation: 0.5-0.7% per year for crystalline silicon"
    },
    {
        "id": "pv_3",
        "text": "Grid-Tied vs Off-Grid Solar Systems:\n\n**Grid-Tied Systems (Most Common)**:\n- Connected to utility grid via net metering\n- Advantages:\n  • Lower cost (no battery storage required)\n  • Sell excess power back to grid\n  • Grid provides backup when solar insufficient\n- Disadvantages:\n  • No power during grid outages (safety requirement)\n  • Subject to utility rate changes and policies\n- Typical ROI: 6-10 years, 15-25% internal rate of return\n\n**Off-Grid Systems (Energy Independence)**:\n- Standalone systems with battery storage\n- Advantages:\n  • Complete energy independence\n  • Power during grid outages\n  • Suitable for remote locations\n- Disadvantages:\n  • Higher upfront cost (+40-60% for batteries)\n  • Requires larger system sizing for reliability\n  • Battery replacement every 10-15 years\n- Typical ROI: 12-20 years, 8-12% internal rate of return\n\n**Hybrid Systems**:\n- Grid-tied with battery backup\n- Best of both worlds: grid interaction + outage protection\n- Cost: Mid-range between grid-tied and off-grid\n- Growing adoption due to grid reliability concerns"
    },
    {
        "id": "pv_4",
        "text": "Solar System Sizing and Economics:\n\n**System Sizing**:\n1. Determine annual consumption (kWh/year from utility bills)\n2. Account for location solar resource (peak sun hours)\n3. Factor system losses (inverter efficiency, soiling, shading): ~20% total\n4. Calculate system size: Annual kWh / (365 × Peak Sun Hours × 0.8)\n\n**Example**: 10,000 kWh/year, 5 peak sun hours\n- System size = 10,000 / (365 × 5 × 0.8) = 6.85 kW\n- Panels needed: 6,850W / 400W = 18 panels\n- Roof area: 18 panels × 2 m² = 36 m² (388 sq ft)\n\n**Economics (2024 US Residential)**:\n- Cost: $2.50-3.50 per watt installed\n- 6.85 kW system: $17,125-23,975\n- Federal tax credit (30% ITC): -$5,138 to -$7,193\n- Net cost: $11,987-16,782\n- Annual savings: $1,200-1,800 (varies by electricity rate)\n- Payback period: 7-14 years\n- 25-year savings: $30,000-45,000 (assuming 3% annual electricity rate increase)\n\n**Financing Options**: Cash purchase, solar loan (5-7% APR), lease ($100-150/month), Power Purchase Agreement (PPA)"
    },
    {
        "id": "pv_5",
        "text": "Solar System Monitoring and Maintenance:\n\n**Monitoring Systems**:\n- Real-time production monitoring via web/mobile app\n- Performance metrics:\n  • Daily/monthly/annual energy production (kWh)\n  • System efficiency vs expected output\n  • Individual panel performance (with microinverters/optimizers)\n  • Environmental impact (CO₂ offset, tree equivalents)\n- Alert notifications for underperformance or faults\n\n**Routine Maintenance**:\n1. **Panel Cleaning**:\n   - Frequency: 2-4 times per year (varies by location)\n   - Method: Soft brush + water (avoid high-pressure washers)\n   - Cost: DIY or $150-300 professional service\n   - Impact: Restores 2-5% production loss from soiling\n\n2. **Visual Inspections** (Annual):\n   - Check for physical damage, cracks, discoloration\n   - Inspect mounting hardware for corrosion/looseness\n   - Verify wire connections and junction boxes\n   - Look for vegetation overgrowth or shading\n\n3. **Electrical Testing** (Every 3-5 years):\n   - Insulation resistance testing\n   - String voltage and current measurements\n   - Inverter performance verification\n   - Grounding system integrity check\n\n**Common Issues**:\n- Inverter failure: Most common issue, typically after 10-15 years\n- Hot spots: Caused by cell defects or shading, detectable via thermal imaging\n- Bypass diode failure: Results in reduced output from shaded panels\n- Bird nesting/pest damage: Requires critter guards or mesh installation"
    }
]

SOLAR_PV_QUERIES = [
    {"id": "pvq1", "text": "What are the main types of solar panels and their efficiencies?", "ref": "The main types are monocrystalline (18-22% efficiency), polycrystalline (15-17%), and thin-film (10-12%)."},
    {"id": "pvq2", "text": "What is the typical lifespan of solar panels?", "ref": "Solar panels typically last 25-30 years with an annual degradation rate of 0.5%."},
    {"id": "pvq3", "text": "What is the difference between string inverters and microinverters?", "ref": "String inverters are centralized and cost-effective ($0.10-0.20/W), while microinverters provide per-panel optimization and higher reliability at $0.40-0.60/W."},
    {"id": "pvq4", "text": "How does temperature affect solar panel performance?", "ref": "Solar panels lose 0.3-0.5% efficiency per degree Celsius above 25°C. A 400W panel at 45°C produces about 368W (8% loss)."},
    {"id": "pvq5", "text": "What are peak sun hours?", "ref": "Peak sun hours are the equivalent hours of 1000 W/m² irradiance per day, typically ranging from 3-4 hours in northern US to 5-7 hours in southwest US."},
    {"id": "pvq6", "text": "How does shading affect solar panel output?", "ref": "Partial shading disproportionately reduces output - 50% shading can cause 80% power loss. Microinverters or power optimizers help mitigate this issue."},
    {"id": "pvq7", "text": "What is the optimal orientation and tilt for solar panels?", "ref": "The optimal orientation is south-facing (northern hemisphere) with tilt angle equal to latitude (±15° acceptable)."},
    {"id": "pvq8", "text": "What is the difference between grid-tied and off-grid solar systems?", "ref": "Grid-tied systems connect to utility grid via net metering (lower cost, no outage power), while off-grid systems use battery storage for complete independence (higher cost, outage protection)."},
    {"id": "pvq9", "text": "What is the typical ROI for residential solar systems?", "ref": "Grid-tied residential solar systems typically have 6-10 year payback periods with 15-25% internal rate of return."},
    {"id": "pvq10", "text": "How do I calculate the size of solar system I need?", "ref": "Calculate system size using: Annual kWh consumption / (365 × Peak Sun Hours × 0.8 efficiency factor)."},
    {"id": "pvq11", "text": "What is the federal solar tax credit?", "ref": "The federal Investment Tax Credit (ITC) provides 30% credit on solar installation costs (as of 2024)."},
    {"id": "pvq12", "text": "How often should solar panels be cleaned?", "ref": "Solar panels should be cleaned 2-4 times per year (varies by location) to restore 2-5% production loss from soiling."},
    {"id": "pvq13", "text": "What are the most common solar system issues?", "ref": "Common issues include inverter failure (after 10-15 years), hot spots from shading, bypass diode failure, and bird nesting/pest damage."},
    {"id": "pvq14", "text": "What is the annual degradation rate for solar panels?", "ref": "Crystalline silicon solar panels degrade at 0.5-0.7% per year."}
]

# ===========================================================================
# Dataset Creation Functions
# ===========================================================================

def create_dataset(name: str, docs: list, queries: list):
    """Create a dataset directory with corpus and queries JSONL files."""

    dataset_dir = Path("datasets") / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write corpus
    corpus_file = dataset_dir / "cases.jsonl"
    with open(corpus_file, 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')

    # Write queries
    queries_file = dataset_dir / "queries.jsonl"
    with open(queries_file, 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')

    print(f" Created {name}:")
    print(f"   Corpus: {len(docs)} documents")
    print(f"   Queries: {len(queries)} questions")
    print(f"   Location: {dataset_dir}/")
    print()

def main():
    print("="*70)
    print("Creating Smart Grid/IoT QA Datasets for XPipe")
    print("="*70)
    print()

    # Create EV/Battery dataset
    create_dataset("ev_battery", EV_BATTERY_DOCS, EV_BATTERY_QUERIES)

    # Create Solar PV dataset
    create_dataset("solar_pv", SOLAR_PV_DOCS, SOLAR_PV_QUERIES)

    print("="*70)
    print(" Dataset Creation Complete!")
    print("="*70)
    print()
    print("New datasets:")
    print("  1. datasets/ev_battery/     (5 docs, 14 queries)")
    print("  2. datasets/solar_pv/       (5 docs, 14 queries)")
    print()
    print("Total datasets now: 7")
    print("  - Network Troubleshooting")
    print("  - Customer Support")
    print("  - ITU Standards")
    print("  - TBMP Legal")
    print("  - Energy/Smart Grid")
    print("  - EV/Battery Management     ← NEW")
    print("  - Solar PV Systems          ← NEW")
    print()
    print("Next steps:")
    print("  1. Update experiments to include new datasets")
    print("  2. Run Experiment 4 with all 7 domains")

if __name__ == '__main__':
    main()
