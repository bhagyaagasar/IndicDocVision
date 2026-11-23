"""Developer helper: renders the sample pages in samples/ (needs playwright + Chrome).
Users do NOT need this; the generated PNGs are committed to the repo.
Add your own Kannada page as samples/sample_kn.png to try Kannada OCR."""
import os
import sys

from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
os.makedirs(OUT, exist_ok=True)

PAGES = {
    "sample_en.png": ("DejaVu Sans, Arial, sans-serif", [
        "<b>City Central Library - Notice</b>",
        "The library is open Monday to Saturday from 9 AM to 5 PM. It is closed on Sundays and public holidays.",
        "Membership fee is Rs 500 per year. Students get a 50 percent discount on showing a valid ID card.",
        "Members can borrow up to four books for fourteen days. A fine of Rs 2 per day is charged for late return.",
    ]),
    "sample_hi.png": ("FreeSerif, 'Noto Sans Devanagari', serif", [
        "<b>नगर केंद्रीय पुस्तकालय - सूचना</b>",
        "पुस्तकालय सोमवार से शनिवार तक सुबह 9 बजे से शाम 5 बजे तक खुला रहता है। रविवार और सार्वजनिक अवकाश के दिन बंद रहता है।",
        "सदस्यता शुल्क 500 रुपये प्रति वर्ष है। वैध पहचान पत्र दिखाने पर विद्यार्थियों को 50 प्रतिशत की छूट मिलती है।",
        "सदस्य चौदह दिनों के लिए अधिकतम चार पुस्तकें ले सकते हैं। देर से लौटाने पर प्रतिदिन 2 रुपये का जुर्माना लगता है।",
    ]),
}

TEMPLATE = """<html><body style="margin:0;background:#fff"><div style="width:1100px;padding:50px 60px;
font:32px/1.7 {font};color:#111;box-sizing:border-box">{body}</div></body></html>"""

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.getenv("CHROME_PATH", "/usr/bin/google-chrome"),
                          args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1100, "height": 700})
    for name, (font, paras) in PAGES.items():
        body = "".join(f"<p style='margin:0 0 18px'>{x}</p>" for x in paras)
        pg.set_content(TEMPLATE.format(font=font, body=body))
        pg.screenshot(path=os.path.join(OUT, name), full_page=True)
        print("wrote", name)
    b.close()
