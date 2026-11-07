"""Generate the PDF and DOCX knowledge-base documents (run once; outputs are committed)."""
from pathlib import Path

from docx import Document as Docx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

KB = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"

# ---------------------------------------------------------------- PDF manual
MANUAL = [
 ("1. Box Contents", [
  "Your AquaPearl 500 Pro is shipped with the following items. Please check the box before you discard the packaging.",
  ("bullets", ["1 x AquaPearl 500 Pro purifier unit with pre-installed filter set (FLT-S5, FLT-C5, FLT-M5)",
   "1 x wall-mount bracket with 4 screws and 4 wall plugs", "1 x inlet tubing, 6 mm, 2 m long",
   "1 x outlet tap and 1/4-inch tap adaptor", "1 x power cable, 1.5 m, 3-pin", "2 x spare tubing clips",
   "1 x quick start card and 1 x warranty card"]),
  "No tools, descaling kit or spare filters are included in the box. The descaling kit (DSC-1) is sold separately."]),
 ("2. Technical Specifications", [
  ("table", [["Specification", "Value"], ["Filtered flow rate", "3.0 L/min"], ["Tank capacity", "8 L"],
   ["Dimensions (W x D x H)", "38 x 18 x 42 cm"], ["Weight", "7.2 kg"], ["Power", "60 W, 220-240 V AC, 50 Hz"],
   ["Inlet water pressure", "15 to 80 psi (1.0 to 5.5 bar)"], ["Maximum inlet TDS", "1,500 ppm"],
   ["Operating temperature", "5 to 38 degrees C"], ["Purification stages", "4: sediment, carbon block, RO membrane, UV-LED"]]),
  "The UV-LED module is rated for 20,000 hours and does not need routine replacement."]),
 ("3. Control Panel and Indicator Lights", [
  "The control panel has a single RESET button, a pairing button and a ring of indicator lights.",
  ("bullets", ["Steady blue: ready to dispense.", "Pulsing blue: the tank is filling.", "Green: UV-LED sterilisation is active.",
   "Amber: attention needed, such as low inlet pressure.", "Red steady or flashing: a filter has reached its limit or a fault has occurred."])]),
 ("4. Operating Instructions", [
  "Open the inlet valve fully, plug in the unit and wait for the ring to turn steady blue. Press the tap lever to dispense water. The unit refills the tank automatically when the level drops.",
  "Do not run the unit with the inlet valve closed. Do not dispense hot water: the AquaPearl 500 Pro supplies room-temperature purified water only."]),
 ("5. Filter Maintenance", [
  "Replace each filter on schedule. The unit counts filter life in the background and alerts you when a filter reaches its limit.",
  ("table", [["Filter", "Part number", "Replace every"], ["Sediment filter (stage 1)", "FLT-S5", "4 months"],
   ["Carbon block filter (stage 2)", "FLT-C5", "8 months"], ["RO membrane (stage 3)", "FLT-M5", "18 months"]]),
  "After replacing a filter, press and hold the RESET button for 5 seconds to reset its counter. After any filter replacement you must flush the system; the flushing procedure is described in the Installation Guide."]),
 ("6. Cleaning and Descaling", [
  "Wipe the exterior with a soft damp cloth once a month. Do not use bleach, solvents or abrasive cleaners.",
  "Descale the unit every 3 months. Sanitise the water tank every 6 months. To descale:",
  ("bullets", ["Switch off the unit, unplug it and close the inlet valve.", "Drain the tank by pressing the tap lever until empty.",
   "Mix one descaling kit (DSC-1) with 1 litre of warm water and pour it into the tank.",
   "Leave the solution for 20 minutes.", "Reopen the inlet valve, plug in the unit and run three full tanks (24 litres) of water to waste.",
   "Resume normal use."])]),
 ("7. Audible Alert Codes", [
  "The unit signals faults with a number of beeps and a light pattern. The codes below apply to firmware 2.4 and later. Units on firmware 2.3 or earlier use an older meaning for E-05; see the Product Changelog.",
  ("table", [["Code", "Beeps and light", "Meaning"],
   ["E-01", "1 beep, amber", "Inlet pressure low or inlet valve closed"], ["E-02", "2 beeps, amber", "Tank-full sensor fault"],
   ["E-03", "3 short beeps, red flashing", "Sediment pre-filter saturated"], ["E-04", "4 beeps, red flashing", "Carbon filter expired"],
   ["E-05", "5 beeps, steady red", "RO membrane saturated (firmware 2.4 and later)"], ["E-06", "6 beeps, red and base light flashing", "Leak detected"],
   ["E-07", "7 beeps, amber flashing", "Pump overheating"], ["E-08", "8 beeps, all lights flashing", "Control board communication fault"]]),
  "For E-03, replace the sediment filter (FLT-S5) and then hold RESET for 5 seconds. For E-06, close the inlet valve, unplug the unit and contact support. For E-08, contact support and do not open the unit."]),
 ("8. Safety Warnings", [
  ("bullets", ["Use only a grounded 3-pin socket and keep the plug dry.", "Never open the housing. Opening the housing by anyone other than a Pearlz-authorised technician is an unauthorised repair.",
   "Install the unit on the cold water line only.", "Keep the unit at least 30 cm away from heat sources.",
   "Switch off and unplug the unit before any maintenance.", "This product is for private household use only."])]),
]


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(40, A4[1] - 28, "Pearlz Home Systems - AquaPearl 500 Pro Product Manual v3.2")
    canvas.drawString(40, 24, "Confidential to customers. Keep for reference.")
    canvas.drawRightString(A4[0] - 40, 24, f"Page {doc.page}")
    canvas.restoreState()


def build_manual() -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10.5, leading=15)
    doc = SimpleDocTemplate(str(KB / "product_manual_aquapearl500pro.pdf"), pagesize=A4,
                            leftMargin=40, rightMargin=40, topMargin=50, bottomMargin=45)
    flow = [Paragraph("AquaPearl 500 Pro Product Manual", styles["Title"]), Paragraph("Version 3.2, March 2026", body), PageBreak()]
    for heading, items in MANUAL:
        flow.append(Paragraph(heading, styles["Heading1"]))
        for item in items:
            if isinstance(item, str):
                flow += [Paragraph(item, body), Spacer(1, 6)]
            elif item[0] == "bullets":
                flow += [Paragraph("- " + b, body) for b in item[1]] + [Spacer(1, 6)]
            else:
                t = Table(item[1], hAlign="LEFT")
                t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                       ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9.5)]))
                flow += [t, Spacer(1, 8)]
        flow.append(PageBreak())
    doc.build(flow[:-1], onFirstPage=_header_footer, onLaterPages=_header_footer)


# --------------------------------------------------------------- DOCX files
INSTALL = [
 ("1. Pre-Installation Checklist", ["Before you start, confirm each item below.",
  "- Water inlet pressure is between 15 and 80 psi. Use a pressure gauge if unsure.",
  "- A grounded 3-pin socket (220-240 V) is within 1.2 metres of the installation spot.",
  "- Clearance: 10 cm at the sides and 30 cm above the unit so that filters can be changed.",
  "- For under-sink installation the cabinet must be at least 45 cm high.",
  "- You will connect to the cold water line only."]),
 ("2. Tools Required", ["Drill with a 6 mm bit, screwdriver, spirit level, tube cutter or scissors, adjustable wrench and PTFE tape."]),
 ("3. Wall-Mount Installation", ["Hold the bracket against the wall and mark the four holes using the spirit level.",
  "Drill the holes, insert the wall plugs and fix the bracket with the four screws.",
  "Hang the unit on the bracket and check that it is level.",
  "Push the inlet tubing into the inlet port until it clicks and secure it with a tubing clip.",
  "Connect the other end to the cold water supply using the tap adaptor and PTFE tape. Fit the outlet tap."]),
 ("4. Under-Sink Installation", ["Place the unit on a flat, dry cabinet floor with the clearances from the checklist.",
  "Connect the inlet tubing to the cold water line and the outlet tap to the sink as in the wall-mount steps.",
  "Route the tubing without kinks and keep the power cable away from water."]),
 ("5. First-Run Flushing Procedure", ["New units must be flushed before the first drink. The new carbon block releases fine carbon dust and the RO membrane is shipped with a preservative.",
  "Open the inlet valve and switch on the unit. Wait for steady blue light. Dispense and discard the first two full tanks, which is 16 litres and takes about 20 minutes. Then dispense a glass and check that the water is clear and has no unusual taste.",
  "The same flushing is needed after you replace filters. After replacing the sediment filter or the carbon filter, discard the first full tank of 8 litres. After replacing the RO membrane, discard the first two full tanks of 16 litres. Skipping the flush is the most common reason for a strange taste or cloudy water right after a filter change. Press and hold RESET for 5 seconds after the replacement and before flushing."]),
 ("6. Professional Installation", ["You can book a Pearlz technician through support@pearlzhome.example. Fees: PKR 2,500 in metro cities, PKR 3,500 in secondary cities and PKR 5,000 in remote areas. The fee is non-refundable once the technician has visited."]),
 ("7. What Voids the Warranty During Self-Installation", ["Following this guide does not affect your warranty. The following do:",
  "- Cutting, shortening or modifying the internal tubing or wiring.",
  "- Using fittings that are not supplied by Pearlz on the inlet.",
  "- Connecting the unit to a hot water line.",
  "- Opening the housing."]),
]

SERVICE = [
 ("1. Purpose", ["This Service and Maintenance Agreement (Pearlz Care) describes the optional annual maintenance plans and the procedure for warranty claims."]),
 ("2. Plans", ["Pearlz Care is sold in three tiers per unit per year.",
  "- Care Basic, PKR 6,000 per year: 1 maintenance visit, technician arrival within 5 working days of a request.",
  "- Care Standard, PKR 11,000 per year: 2 maintenance visits, technician arrival within 3 working days.",
  "- Care Premium, PKR 18,000 per year: 4 maintenance visits, technician arrival within 1 working day, and one full filter set (FLT-S5, FLT-C5 and FLT-M5) included each year."]),
 ("3. Parts and Labour", ["Labour for visits included in the plan, O-rings and tubing replaced during a visit are covered. Filters are billed at the list price in the Pricing Guide, except where the Premium plan includes them. Parts outside the warranty are billed at list price. A visit outside a plan costs PKR 2,000 labour. A technical assessment of an out-of-warranty unit costs PKR 1,500, which is waived if you accept the repair. Visits to remote areas carry a travel surcharge of PKR 3,000."]),
 ("4. Cancellation and Pro-Rata Refund", ["You may cancel within 14 days of purchase for a full refund if no visit has taken place. After that, we refund the unused full months, minus PKR 2,000 for each visit already used, minus a 10% administration fee."]),
 ("5. Warranty Claim Procedure", ["This is the claim procedure referred to in the Warranty Policy.",
  "Step 1: Contact support@pearlzhome.example within 7 days of noticing the defect. Provide the invoice, the unit serial number, and photos or a short video of the problem.",
  "Step 2: Pearlz issues a ticket number within 1 working day.",
  "Step 3: Support runs a remote diagnosis. Some faults are fixed by remote guidance.",
  "Step 4: If a visit is needed, a Pearlz-authorised technician attends within 5 working days in metro and secondary cities and within 10 working days in remote areas.",
  "Step 5: Pearlz repairs or replaces the covered part. If the unit cannot be repaired within 15 working days of the first visit, Pearlz replaces the unit.",
  "Claims for items excluded by the Warranty Policy, such as filters or damage to property, are declined, and the customer may pay for a service visit instead."]),
 ("6. Leaks", ["If a unit leaks, close the inlet valve, unplug the unit and report it immediately. Pearlz repairs or replaces covered parts of the unit only, in line with the Warranty Policy."]),
]


def build_docx(path: str, title: str, sections) -> None:
    d = Docx()
    d.add_heading(title, 0)
    for heading, paras in sections:
        d.add_heading(heading, 1)
        for p in paras:
            d.add_paragraph(p)
    d.save(str(KB / path))


if __name__ == "__main__":
    build_manual()
    build_docx("installation_guide.docx", "Installation Guide", INSTALL)
    build_docx("service_maintenance_agreement.docx", "Service and Maintenance Agreement", SERVICE)
    print("generated")
