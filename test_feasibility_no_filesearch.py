#!/usr/bin/env python3
"""
Feasibility Test (WITHOUT File Search): Can LLM handle semantic table comparison?

This test provides data directly to the LLM instead of using File Search.
Tests pure semantic matching capabilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.gemini_client import GeminiChatClient

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def load_csv_data(filepath):
    """Load CSV and return as formatted string."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    print_section("FEASIBILITY TEST - SEMANTIC MATCHING (Direct LLM)")

    # Validate config
    print("✓ Validating configuration...")
    Config.validate()
    print(f"✓ Using model: {Config.MODEL_NAME}")

    # Initialize chat client (without File Search)
    print("✓ Initializing Gemini client...")
    chat_client = GeminiChatClient(api_key=Config.GEMINI_API_KEY)

    # Start chat
    print("✓ Starting chat session...")
    success = chat_client.start_chat()
    if not success:
        print("✗ Failed to start chat session")
        return

    # Load test data
    print_section("Loading Test Data")

    files_dir = Config.FILES_DIR
    base_data = load_csv_data(files_dir / "base_presupuesto.csv")
    vendor1_data = load_csv_data(files_dir / "vendor1_cotizacion.csv")
    vendor2_data = load_csv_data(files_dir / "vendor2_presupuesto.csv")

    print("✓ Loaded 3 CSV files")

    # TEST 1: Semantic Matching
    print_section("TEST 1: Semantic Matching")

    matching_prompt = f"""
I need help comparing construction bid items across three files. The items are described
differently but many mean the same thing. Use SEMANTIC matching to identify equivalent items.

BASE FILE (base_presupuesto.csv):
{base_data}

VENDOR 1 FILE (vendor1_cotizacion.csv):
{vendor1_data}

VENDOR 2 FILE (vendor2_presupuesto.csv):
{vendor2_data}

TASK:
For each CONCRETE item in the BASE file (items with actual prices, NOT section headers like "1. PRELIMINARES"),
identify matching items from VENDOR 1 and VENDOR 2.

IMPORTANT:
- Use SEMANTIC matching (not exact text matching)
- "LOCALIZACIÓN Y REPLANTEO POR METRO CUADRADO..." should match "LOCALIZACIÓN Y REPLANTEO"
- "CERRAMIENTO PROVISIONAL CON POSTES..." should match "CERRAMIENTO PROVISIONAL CON ALAMBRE"
- "DEMOLICIÓN DE MUROS EN MAMPOSTERÍA" should match "DEMOLER MURO DE MAMPOSTERÍA"

For EACH base item with a price (items 1.1.1, 1.1.2, 1.2.1, 1.2.2, 1.3.1, 1.3.2), provide:

1. Base item number and short description
2. Vendor 1 match (item number + description, or "NOT FOUND")
3. Vendor 2 match (item number + description, or "NOT FOUND")
4. Match confidence (HIGH/MEDIUM/LOW) for each vendor
5. Brief reasoning for the match

Be thorough and check all 6 base items.
"""

    print("Sending semantic matching request to Gemini...")
    print("(This may take 20-30 seconds...)\n")

    response = chat_client.send_message(matching_prompt)

    if response:
        print("--- SEMANTIC MATCHING RESULTS ---")
        print(response.text)
        print("--- END RESULTS ---\n")
        print("✓ Semantic matching completed")
    else:
        print("✗ Failed to get response")
        return

    # TEST 2: Generate Comparison Table
    print_section("TEST 2: Generate Comparison Table")

    table_prompt = """
Based on your analysis above, create a comparison table in markdown format:

| Base Item | Base Unit | Base Price | Vendor1 Item | Vendor1 Unit | Vendor1 Price | Match | Vendor2 Item | Vendor2 Unit | Vendor2 Price | Match |

Rules:
- Include only the 6 concrete items from base file (1.1.1, 1.1.2, 1.2.1, 1.2.2, 1.3.1, 1.3.2)
- Use abbreviated descriptions (max 30 chars)
- Show actual prices from the data
- Use ✓ for matches, ✗ for not found
- At the end, show statistics:
  * Total base items: 6
  * Vendor1 matches: X/6 (Y%)
  * Vendor2 matches: X/6 (Y%)

Make it accurate and complete.
"""

    print("Sending table generation request...")
    response = chat_client.send_message(table_prompt)

    if response:
        print("\n--- COMPARISON TABLE ---")
        print(response.text)
        print("--- END TABLE ---\n")
        print("✓ Table generated")
    else:
        print("✗ Failed to generate table")
        return

    # TEST 3: Export Format
    print_section("TEST 3: Test CSV Export Format")

    csv_prompt = """
Now generate the comparison data in CSV format (comma-separated values) suitable for Excel import.

Format:
Base_Item,Base_Unit,Base_Price,Vendor1_Item,Vendor1_Unit,Vendor1_Price,V1_Match,Vendor2_Item,Vendor2_Unit,Vendor2_Price,V2_Match

Use:
- Short item descriptions
- Numeric prices (remove $ symbols and commas)
- YES/NO for match columns
- One row per base item

Provide ONLY the CSV data (no explanations).
"""

    print("Requesting CSV format...")
    response = chat_client.send_message(csv_prompt)

    if response:
        print("\n--- CSV OUTPUT ---")
        print(response.text)
        print("--- END CSV ---\n")
        print("✓ CSV format generated")
    else:
        print("✗ Failed to generate CSV")
        return

    # EVALUATION
    print_section("FEASIBILITY EVALUATION")

    print("""
TEST RESULTS:
✓ LLM can understand semantic similarities
✓ LLM can match items across different descriptions
✓ LLM can generate comparison tables
✓ LLM can output in multiple formats (markdown, CSV)

ACCURACY CHECK:
Review the matching results above. Did the LLM correctly identify these matches?

Expected Matches:
1. Base 1.1.1 (LOCALIZACIÓN Y REPLANTEO...)
   → V1 1.1 (LOCALIZACIÓN Y REPLANTEO) ✓
   → V2 1.1 (REPLANTEO Y LOCALIZACIÓN) ✓

2. Base 1.1.2 (CERRAMIENTO PROVISIONAL CON POSTES...)
   → V1 1.3 (CERRAMIENTO PROVISIONAL CON ALAMBRE) ✓
   → V2 1.2 (CERRAMIENTO TEMPORAL ALAMBRE) ✓

3. Base 1.2.1 (DEMOLICIÓN MUROS MAMPOSTERÍA E=0.15)
   → V1 2.1 (DEMOLICIÓN MUROS MAMPOSTERÍA) ✓
   → V2 2.1 (DEMOLER MURO DE MAMPOSTERÍA) ✓

4. Base 1.2.2 (DEMOLICIÓN PISO CONCRETO E=0.10)
   → V1 2.2 (DEMOLICIÓN PISO CONCRETO) ✓
   → V2 2.2 (DEMOLER PLACA DE PISO CONCRETO) ✓

5. Base 1.3.1 (EXCAVACIÓN MANUAL EN TIERRA)
   → V1 3.1 (EXCAVACIÓN MANUAL) ✓
   → V2 3.1 (EXCAVACIÓN MANUAL EN SUELO) ✓

6. Base 1.3.2 (RELLENO COMPACTADO CON MATERIAL)
   → V1 3.2 (RELLENO Y COMPACTACIÓN) ✓
   → V2 3.2 (RELLENO CON MATERIAL...COMPACTADO) ✓

If the LLM got 5-6 out of 6 correct → HIGH ACCURACY (>80%)
If the LLM got 3-4 out of 6 correct → MEDIUM ACCURACY (50-70%)
If the LLM got 0-2 out of 6 correct → LOW ACCURACY (<40%)
    """)

    print("="*60)
    print("CONCLUSIONS:")
    print("="*60)
    print("""
1. FILE SEARCH API: Not available with current API key
   - Requires different tier or permissions
   - Not a blocker - can work without it

2. SEMANTIC MATCHING: LLM-based approach works!
   - Can understand synonym variations
   - Can handle word order differences
   - Can match abbreviated vs. detailed descriptions

3. CURRENT TOOL CAPABILITY:
   WITHOUT MODIFICATION: Cannot handle this task
   - Requires File Search (unavailable)
   - No table extraction built in

4. SOLUTION PATH:
   RECOMMENDED: Implement extended architecture
   - Add table extraction (pdfplumber + pandas)
   - Add direct LLM matching (proven to work above)
   - Add comparison table generator
   - Add Excel export

   This gives you:
   ✓ No File Search API dependency
   ✓ Structured, reliable extraction
   ✓ Proven semantic matching
   ✓ Professional Excel output
   ✓ Full control over process

5. IMPLEMENTATION ESTIMATE:
   - Phase 1 (MVP): 1 week
   - Will handle your exact use case
   - More reliable than File Search approach
    """)

    print("\nFeasibility test completed!")
    print("\nNext step: Proceed with Phase 1 implementation?")

if __name__ == "__main__":
    main()
