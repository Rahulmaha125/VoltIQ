import streamlit as st
import numpy as np
import pickle
import io
import plotly.express as px
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Devanagari TTF Font for ReportLab if available
try:
    pdfmetrics.registerFont(TTFont('Nirmala', 'C:/Windows/Fonts/Nirmala.ttc', subfontIndex=0))
    PDF_FONT = 'Nirmala'
except Exception:
    PDF_FONT = 'Helvetica'

# Set page configuration
st.set_page_config(page_title="Electricity & Solar Predictor Pro", page_icon="⚡", layout="wide")

# Function to make prediction using the saved model and scaler
def predict_electricity_consumption(input_features):
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
        
    input_array = np.array(input_features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    return max(0.5, prediction[0])

# Multilingual PDF Translation Dictionary
PDF_TRANSLATIONS = {
    'English': {
        'title': '⚡ ELECTRICITY CONSUMPTION & SOLAR AUDIT STATEMENT',
        'subtitle': 'Official System-Generated Energy Statement & Sustainability Report',
        'stmt_id': 'Statement ID',
        'date': 'Date',
        'area': 'Property Area',
        'occupants': 'Occupants',
        'temp': 'Ambient Temp',
        'ac_hours': 'AC Usage',
        'tariff_rate': 'Tariff Rate',
        'weekend': 'Weekend Flag',
        'sec1_title': '📊 1. Monthly Electricity Bill Breakdown',
        'particulars': 'Particulars',
        'val': 'Calculated Value',
        'daily_kwh': 'Daily Energy Consumption',
        'monthly_kwh': 'Monthly Energy Consumed (30 Days)',
        'applicable_rate': 'Applicable Tariff Rate',
        'monthly_bill': 'ESTIMATED MONTHLY ELECTRICITY BILL',
        'yearly_bill': 'Estimated Annual Electricity Expense',
        'sec2_title': '🍃 2. Carbon Footprint & Environmental Impact',
        'env_metric': 'Environmental Metric',
        'impact_level': 'Impact Level',
        'monthly_co2': 'Monthly CO₂ Grid Emissions',
        'annual_co2': 'Annual CO₂ Grid Emissions',
        'trees_saved': 'Equivalent Trees Saved per Year with Solar',
        'sec3_title': '☀️ 3. Solar Panel Investment & Payback Analysis',
        'solar_param': 'Solar Parameter',
        'solar_val': 'Value / Estimate',
        'rec_capacity': 'Recommended Solar Capacity',
        'gross_cost': 'Estimated Gross Solar System Cost',
        'subsidy': 'Est. Govt Subsidy (PM Surya Ghar)',
        'net_cost': 'Net Capital Outlay (After Subsidy)',
        'payback': 'Estimated Payback Period',
        'solar_roi': 'Annual Return on Investment (ROI)',
        'footer': 'Powered by Rahul Mahanavar Software',
        'btn_label': '📥 Download PDF Statement (English)'
    },
    'मराठी (Marathi)': {
        'title': '⚡ वीज वापर आणि सौर ऊर्जा तपासणी अहवाल',
        'subtitle': 'अधिकृत सिस्टीम-जनरेटेड ऊर्जा विवरणपत्र आणि पर्यावरण अहवाल',
        'stmt_id': 'अहवाल आयडी',
        'date': 'दिनांक',
        'area': 'इमारतीचे क्षेत्रफळ',
        'occupants': 'घरातील व्यक्ती',
        'temp': 'तापमान',
        'ac_hours': 'एसी वापर',
        'tariff_rate': 'विजेचा दर',
        'weekend': 'वीकेंड स्थिती',
        'sec1_title': '📊 १. मासिक वीज बिल तपशील',
        'particulars': 'तपशील',
        'val': 'गणित मूल्य',
        'daily_kwh': 'दैनंदिन वीज वापर',
        'monthly_kwh': 'मासिक एकूण वीज वापर (३० दिवस)',
        'applicable_rate': 'लागू विजेचा दर',
        'monthly_bill': 'अंदाजे मासिक वीज बिल',
        'yearly_bill': 'अंदाजे वार्षिक वीज खर्च',
        'sec2_title': '🍃 २. कार्बन फूटप्रिंट आणि पर्यावरण परिणाम',
        'env_metric': 'पर्यावरणीय निर्देशांक',
        'impact_level': 'परिणाम पातळी',
        'monthly_co2': 'मासिक CO₂ कार्बन उत्सर्जन',
        'annual_co2': 'वार्षिक CO₂ कार्बन उत्सर्जन',
        'trees_saved': 'सोलाअरमुळे वाचणारी वार्षिक झाडे',
        'sec3_title': '☀️ ३. सौर ऊर्जा गुंतवणूक आणि परतावा विश्लेषण',
        'solar_param': 'सौर ऊर्जा घटक',
        'solar_val': 'मूल्य / अंदाज',
        'rec_capacity': 'शिफारस केलेली सौर ऊर्जा क्षमता',
        'gross_cost': 'एकूण सोलाअर सिस्टीम खर्च',
        'subsidy': 'सरकारी सबसिडी (PM Surya Ghar)',
        'net_cost': 'निव्वळ गुंतवणूक (सबसिडीनंतर)',
        'payback': 'अंदाजे मुद्दल वसुली कालावधी',
        'solar_roi': 'वार्षिक गुंतवणुकीवरील परतावा (ROI)',
        'footer': 'Powered by Rahul Mahanavar Software',
        'btn_label': '📥 डाऊनलोड करा PDF विवरणपत्र (मराठी)'
    },
    'हिंदी (Hindi)': {
        'title': '⚡ बिजली खपत और सौर ऊर्जा लेखा परीक्षा विवरण',
        'subtitle': 'आधिकारिक प्रणाली-जनरेटेड ऊर्जा विवरण और पर्यावरण रिपोर्ट',
        'stmt_id': 'विवरण आईडी',
        'date': 'दिनांक',
        'area': 'भवन का क्षेत्रफल',
        'occupants': 'रहने वाले लोग',
        'temp': 'तापमान',
        'ac_hours': 'एसी उपयोग',
        'tariff_rate': 'बिजली की दर',
        'weekend': 'वीकेंड स्थिति',
        'sec1_title': '📊 १. मासिक बिजली बिल विवरण',
        'particulars': 'विवरण',
        'val': 'गणना मूल्य',
        'daily_kwh': 'दैनिक बिजली खपत',
        'monthly_kwh': 'मासिक कुल बिजली खपत (३० दिन)',
        'applicable_rate': 'लागू बिजली दर',
        'monthly_bill': 'अनुमानित मासिक बिजली बिल',
        'yearly_bill': 'अनुमानित वार्षिक बिजली खर्च',
        'sec2_title': '🍃 २. कार्बन पदचिह्न और पर्यावरण प्रभाव',
        'env_metric': 'पर्यावरण संकेतक',
        'impact_level': 'प्रभाव स्तर',
        'monthly_co2': 'मासिक CO₂ उत्सर्जन',
        'annual_co2': 'वार्षिक CO₂ उत्सर्जन',
        'trees_saved': 'सोलर से प्रतिवर्ष बचाए गए पेड़',
        'sec3_title': '☀️ ३. सौर ऊर्जा निवेश और वापसी विश्लेषण',
        'solar_param': 'सौर ऊर्जा पैरामीटर',
        'solar_val': 'मूल्य / अनुमान',
        'rec_capacity': 'अनुशंसित सौर ऊर्जा क्षमता',
        'gross_cost': 'कुल सोलर सिस्टम लागत',
        'subsidy': 'सरकारी सब्सिडी (PM Surya Ghar)',
        'net_cost': 'शुद्ध पूंजी निवेश (सब्सिडी के बाद)',
        'payback': 'अनुमानित लागत वापसी अवधि',
        'solar_roi': 'वार्षिक निवेश पर प्रतिफल (ROI)',
        'footer': 'Powered by Rahul Mahanavar Software',
        'btn_label': '📥 डाउनलोड करें PDF विवरण (हिंदी)'
    }
}

# Function to generate PDF Electricity Bill & Energy Audit Statement in Selected Language
def generate_pdf_report(data, lang='English'):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    t = PDF_TRANSLATIONS.get(lang, PDF_TRANSLATIONS['English'])
    
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20,
        textColor=colors.HexColor('#1E3A8A'), alignment=1, fontName=PDF_FONT
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle', parent=styles['Normal'], fontSize=9, leading=11,
        textColor=colors.HexColor('#4B5563'), alignment=1, fontName=PDF_FONT
    )
    section_heading = ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'], fontSize=11, leading=14,
        textColor=colors.HexColor('#1E3A8A'), fontName=PDF_FONT, spaceBefore=8, spaceAfter=4
    )
    normal_style = ParagraphStyle(
        'NormalStyle', parent=styles['Normal'], fontSize=8.5, leading=11,
        textColor=colors.HexColor('#1F2937'), fontName=PDF_FONT
    )
    bold_style = ParagraphStyle('BoldStyle', parent=normal_style, fontName=PDF_FONT)

    elements = []
    
    # Header Banner
    elements.append(Paragraph(f"<b>{t['title']}</b>", title_style))
    elements.append(Paragraph(t['subtitle'], subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceBefore=1, spaceAfter=8))

    # Statement & Consumer Metadata Table
    meta_data = [
        [Paragraph(f"<b>{t['stmt_id']}:</b> EA-2026-8842", normal_style), Paragraph(f"<b>{t['date']}:</b> 23 Sep 2026", normal_style)],
        [Paragraph(f"<b>{t['area']}:</b> {data['area']} sq ft", normal_style), Paragraph(f"<b>{t['occupants']}:</b> {data['occupants']}", normal_style)],
        [Paragraph(f"<b>{t['temp']}:</b> {data['temp']} °C", normal_style), Paragraph(f"<b>{t['ac_hours']}:</b> {data['ac_hours']} Hours/Day", normal_style)],
        [Paragraph(f"<b>{t['tariff_rate']}:</b> ₹ {data['rate']:.2f} / kWh", normal_style), Paragraph(f"<b>{t['weekend']}:</b> {'Yes' if data['is_weekend']==1 else 'No'}", normal_style)]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # Section 1: Monthly Bill Breakdown (Electricity Utility Bill Style)
    elements.append(Paragraph(f"<b>{t['sec1_title']}</b>", section_heading))
    bill_data = [
        [Paragraph(f"<b>{t['particulars']}</b>", bold_style), Paragraph(f"<b>{t['val']}</b>", bold_style)],
        [Paragraph(t['daily_kwh'], normal_style), Paragraph(f"{data['daily_kwh']:.2f} kWh", normal_style)],
        [Paragraph(t['monthly_kwh'], normal_style), Paragraph(f"{data['monthly_kwh']:.1f} Units (kWh)", normal_style)],
        [Paragraph(t['applicable_rate'], normal_style), Paragraph(f"₹ {data['rate']:.2f} / kWh", normal_style)],
        [Paragraph(f"<b>{t['monthly_bill']}</b>", bold_style), Paragraph(f"<b>₹ {data['monthly_bill']:,.2f}</b>", bold_style)],
        [Paragraph(t['yearly_bill'], normal_style), Paragraph(f"₹ {data['yearly_bill']:,.2f}", normal_style)]
    ]
    bill_table = Table(bill_data, colWidths=[350, 190])
    bill_table.setStyle(TableStyle([
        ('HEADERBACKGROUND', (0,0), (-1,0), colors.HexColor('#DBEAFE')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#FEF08A')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(bill_table)
    elements.append(Spacer(1, 10))

    # Section 2: Carbon Footprint & Environmental Impact
    elements.append(Paragraph(f"<b>{t['sec2_title']}</b>", section_heading))
    carbon_data = [
        [Paragraph(f"<b>{t['env_metric']}</b>", bold_style), Paragraph(f"<b>{t['impact_level']}</b>", bold_style)],
        [Paragraph(t['monthly_co2'], normal_style), Paragraph(f"{data['monthly_co2']:.1f} kg CO₂", normal_style)],
        [Paragraph(t['annual_co2'], normal_style), Paragraph(f"{data['annual_co2']:.1f} kg CO₂", normal_style)],
        [Paragraph(f"<b>{t['trees_saved']}</b>", bold_style), Paragraph(f"<b>{data['trees_saved']} Trees 🌳</b>", bold_style)]
    ]
    carbon_table = Table(carbon_data, colWidths=[350, 190])
    carbon_table.setStyle(TableStyle([
        ('HEADERBACKGROUND', (0,0), (-1,0), colors.HexColor('#DCFCE7')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#F0FDF4')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(carbon_table)
    elements.append(Spacer(1, 10))

    # Section 3: Solar Investment & Payback Roadmap
    elements.append(Paragraph(f"<b>{t['sec3_title']}</b>", section_heading))
    solar_data = [
        [Paragraph(f"<b>{t['solar_param']}</b>", bold_style), Paragraph(f"<b>{t['solar_val']}</b>", bold_style)],
        [Paragraph(t['rec_capacity'], normal_style), Paragraph(f"<b>{data['solar_kw']} kW</b>", normal_style)],
        [Paragraph(t['gross_cost'], normal_style), Paragraph(f"₹ {data['gross_solar_cost']:,.2f}", normal_style)],
        [Paragraph(t['subsidy'], normal_style), Paragraph(f"- ₹ {data['subsidy']:,.2f}", normal_style)],
        [Paragraph(f"<b>{t['net_cost']}</b>", bold_style), Paragraph(f"<b>₹ {data['net_solar_cost']:,.2f}</b>", bold_style)],
        [Paragraph(f"<b>{t['payback']}</b>", bold_style), Paragraph(f"<b>~ {data['payback_years']:.1f} Years ⏳</b>", bold_style)],
        [Paragraph(t['solar_roi'], normal_style), Paragraph(f"{data['solar_roi']:.1f} % / yr", normal_style)]
    ]
    solar_table = Table(solar_data, colWidths=[350, 190])
    solar_table.setStyle(TableStyle([
        ('HEADERBACKGROUND', (0,0), (-1,0), colors.HexColor('#FFEDD5')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#FFEDD5')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(solar_table)
    elements.append(Spacer(1, 12))

    # Footer - Powered by Rahul Mahanavar Software
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#94A3B8'), spaceBefore=5, spaceAfter=5))
    elements.append(Paragraph(f"<b>{t['footer']}</b>", subtitle_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit App UI Header
st.title('⚡ Electricity Consumption, Bill & Solar Predictor Pro')
st.markdown('Predict daily electricity usage (kWh), estimate monthly bill (₹), evaluate carbon footprint, and analyze Solar ROI & Payback period!')

# Sidebar for User Input
st.sidebar.header('⚙️ Input Features')

temp = st.sidebar.number_input('Temperature (°C)', min_value=0.0, max_value=60.0, value=25.0, step=0.5)
humidity = st.sidebar.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
area = st.sidebar.number_input('Building Area (sq ft)', min_value=100.0, max_value=10000.0, value=800.0, step=50.0)
occupants = st.sidebar.number_input('Occupants Count', min_value=1, max_value=50, value=4, step=1)
appliances = st.sidebar.number_input('Appliance Count', min_value=1, max_value=50, value=5, step=1)
is_weekend = st.sidebar.selectbox('Is Weekend?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
ac_hours = st.sidebar.number_input('AC Usage Hours (Hours/Day)', min_value=0.0, max_value=24.0, value=0.0, step=0.5)

st.sidebar.markdown('---')
st.sidebar.header('💵 Electricity Tariff Settings')
unit_rate = st.sidebar.number_input('Electricity Rate (₹ / kWh)', min_value=1.0, max_value=25.0, value=8.0, step=0.5)

input_features = [temp, humidity, area, occupants, appliances, is_weekend, ac_hours]

# Initialize Session State for persistent prediction across widget interactions
if 'has_predicted' not in st.session_state:
    st.session_state['has_predicted'] = False

# Trigger prediction when button is clicked
if st.button('🚀 Predict & Calculate Full Impact', type='primary'):
    st.session_state['has_predicted'] = True

if st.session_state['has_predicted']:
    daily_kwh = predict_electricity_consumption(input_features)
    monthly_kwh = daily_kwh * 30.0
    monthly_bill = monthly_kwh * unit_rate
    yearly_bill = monthly_bill * 12.0
    
    # Solar capacity calculation: 1 kW solar generates ~4 kWh per day
    recommended_solar_kw = max(1.0, round(daily_kwh / 4.0, 1))
    
    # 1. Carbon Footprint Calculations (0.82 kg CO2 per kWh)
    monthly_co2 = monthly_kwh * 0.82
    annual_co2 = monthly_co2 * 12.0
    trees_saved = int(annual_co2 / 20.0)
    
    # 2. Solar ROI & Payback Calculations
    gross_solar_cost = recommended_solar_kw * 50000.0
    subsidy = min(78000.0, 30000.0 * min(2, recommended_solar_kw) + (18000.0 if recommended_solar_kw >= 3 else 0.0))
    net_solar_cost = max(10000.0, gross_solar_cost - subsidy)
    payback_years = round(net_solar_cost / yearly_bill, 1) if yearly_bill > 0 else 0.0
    solar_roi = round((yearly_bill / net_solar_cost) * 100.0, 1) if net_solar_cost > 0 else 0.0
    
    # 3. Seasonal Comparison Predictions
    summer_kwh = predict_electricity_consumption([min(50.0, temp + 6.0), humidity, area, occupants, appliances, is_weekend, min(24.0, ac_hours + 3.0)])
    summer_bill = summer_kwh * 30.0 * unit_rate
    
    winter_kwh = predict_electricity_consumption([max(15.0, temp - 5.0), humidity, area, occupants, appliances, is_weekend, 0.0])
    winter_bill = winter_kwh * 30.0 * unit_rate
    
    st.markdown("---")
    st.subheader("📊 1. Primary Prediction & Energy Insights")
    
    # Display 4 Main Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Daily Consumption", value=f"{daily_kwh:.2f} kWh")
    with c2:
        st.metric(label="Monthly Units", value=f"{monthly_kwh:.1f} Units")
    with c3:
        st.metric(label="Estimated Monthly Bill", value=f"₹ {monthly_bill:,.2f}")
    with c4:
        st.metric(label="Recommended Solar Size", value=f"{recommended_solar_kw} kW")
        
    st.markdown("---")
    
    # Detailed Impact Cards Row (Bill & Solar & Carbon)
    r1_col1, r1_col2 = st.columns(2)
    
    with r1_col1:
        st.info("### 💵 Monthly Bill & Tariff Breakdown\n"
                f"- **Daily Energy Used**: `{daily_kwh:.2f} kWh`\n"
                f"- **Monthly Energy Consumed**: `{monthly_kwh:.2f} kWh`\n"
                f"- **Tariff Rate**: `₹ {unit_rate:.2f} / kWh`\n"
                f"- **Estimated Monthly Bill**: **`₹ {monthly_bill:,.2f}`**\n"
                f"- **Estimated Yearly Bill**: **`₹ {yearly_bill:,.2f}`**\n")

    with r1_col2:
        st.success("### ☀️ Solar ROI & Investment Roadmap\n"
                   f"- **Recommended Capacity**: **`{recommended_solar_kw} kW`**\n"
                   f"- **Gross Solar Cost**: `₹ {gross_solar_cost:,.2f}`\n"
                   f"- **Est. Govt Subsidy (PM Surya Ghar)**: `- ₹ {subsidy:,.2f}`\n"
                   f"- **Net Investment (After Subsidy)**: **`₹ {net_solar_cost:,.2f}`**\n"
                   f"- **Payback Period**: **`~ {payback_years} Years ⏳`** *(Annual ROI: {solar_roi}%/yr)*\n")

    st.markdown("---")
    
    # Feature 1 & Feature 3 Row: Carbon Footprint & Seasonal Chart
    r2_col1, r2_col2 = st.columns(2)
    
    with r2_col1:
        st.subheader("🍃 Carbon Footprint & Environmental Impact")
        st.warning(
            f"- 🏭 **Monthly CO₂ Grid Emissions**: `{monthly_co2:.1f} kg CO₂`\n"
            f"- 💨 **Annual CO₂ Grid Emissions**: `{annual_co2:.1f} kg CO₂`\n"
            f"- 🌳 **Trees Saved Equivalent with Solar**: **`{trees_saved} Trees Saved / Year`**\n\n"
            f"💡 *Installing a `{recommended_solar_kw} kW` Solar System reduces your carbon footprint to 0 and saves {trees_saved} trees every year!*"
        )
        
    with r2_col2:
        st.subheader("☀️ ⛅ ❄️ Seasonal Bill Comparison")
        seasonal_df = pd.DataFrame({
            'Season': ['Summer (उन्हाळा)', 'Monsoon (पावसाळा/Current)', 'Winter (हिवाळा)'],
            'Monthly Bill (₹)': [round(summer_bill, 2), round(monthly_bill, 2), round(winter_bill, 2)]
        })
        fig = px.bar(
            seasonal_df, x='Season', y='Monthly Bill (₹)', text='Monthly Bill (₹)',
            color='Season', color_discrete_sequence=['#EF4444', '#3B82F6', '#10B981'],
            title="Estimated Bill Variation Across Seasons"
        )
        fig.update_traces(texttemplate='₹ %{text:,.2f}', textposition='outside')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Feature 4: Downloadable Multilingual PDF Electricity Bill Report
    st.subheader("📄 Download Official Electricity & Solar Audit Statement (PDF)")
    
    col_pdf1, col_pdf2 = st.columns([1, 2])
    with col_pdf1:
        selected_pdf_lang = st.selectbox(
            "🌐 Select PDF Language / भाषा निवडा",
            ['English', 'मराठी (Marathi)', 'हिंदी (Hindi)'],
            index=0,
            key='pdf_lang_select'
        )
    
    report_data = {
        'area': area,
        'occupants': occupants,
        'temp': temp,
        'ac_hours': ac_hours,
        'rate': unit_rate,
        'is_weekend': is_weekend,
        'daily_kwh': daily_kwh,
        'monthly_kwh': monthly_kwh,
        'monthly_bill': monthly_bill,
        'yearly_bill': yearly_bill,
        'monthly_co2': monthly_co2,
        'annual_co2': annual_co2,
        'trees_saved': trees_saved,
        'solar_kw': recommended_solar_kw,
        'gross_solar_cost': gross_solar_cost,
        'subsidy': subsidy,
        'net_solar_cost': net_solar_cost,
        'payback_years': payback_years,
        'solar_roi': solar_roi
    }
    
    pdf_bytes = generate_pdf_report(report_data, lang=selected_pdf_lang)
    btn_label = PDF_TRANSLATIONS.get(selected_pdf_lang, {}).get('btn_label', '📥 Download PDF Statement')
    
    with col_pdf2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label=btn_label,
            data=pdf_bytes,
            file_name=f"Electricity_Bill_and_Solar_Audit_Report_{selected_pdf_lang.split()[0]}.pdf",
            mime="application/pdf",
            type="primary"
        )
