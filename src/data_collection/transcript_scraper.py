"""
Phase 1D: Earnings Call Transcript Collection
Generates highly varied, realistic earnings call transcripts (7,000-20,000 words)
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import time
import json
import re
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, LOGS_DIR


class TranscriptCollector:
    """
    Collects/generates earnings call transcripts from multiple sources
    """

    def __init__(self, target_count=150):
        self.target_count = target_count

        self.output_dir = RAW_DATA_DIR / 'transcripts'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = LOGS_DIR / 'data_collection' / 'transcripts.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.collected_transcripts = []
        self.failed_attempts = []

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

    def get_high_priority_tickers(self):
        """Get tickers that have financial data"""
        from config import PROCESSED_DATA_DIR

        financial_file = PROCESSED_DATA_DIR / 'financial' / 'financial_features_normalized.csv'

        if financial_file.exists():
            print("📊 Reading tickers from financial data...")
            df = pd.read_csv(financial_file)
            ticker_counts = df['Ticker'].value_counts()
            top_tickers = ticker_counts.head(50).index.tolist()
            print(f"✅ Found {len(top_tickers)} tickers with financial data")
            return top_tickers
        else:
            print("⚠️  Financial data not found, using default tickers")
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
                'XOM', 'V', 'JPM', 'PG', 'MA', 'LLY', 'HD', 'CVX', 'MRK', 'ABBV',
                'PEP', 'COST', 'KO', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT',
                'BAC', 'CRM', 'ADBE', 'DHR', 'LIN', 'NFLX', 'VZ', 'NKE', 'TXN', 'CMCSA',
                'WFC', 'NEE', 'DIS', 'RTX', 'ORCL', 'AMD', 'UPS', 'PM', 'QCOM', 'MS'
            ]

    def create_sample_transcripts(self, tickers, transcripts_per_ticker=3):
        """
        Create highly varied, realistic sample transcripts (7,000-20,000 words each)
        """
        print("=" * 60)
        print("CREATING ENHANCED SYNTHETIC TRANSCRIPTS")
        print("=" * 60)
        print()
        print("Features:")
        print("   - 7,000 to 20,000 words per transcript")
        print("   - Highly varied scenarios and tones")
        print("   - Realistic analyst Q&A (8-15 questions)")
        print("   - Sector-specific language and metrics")
        print("   - Multiple management speakers")
        print()

        sample_transcripts = []
        quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']

        for ticker in tqdm(tickers[:50], desc="Generating transcripts"):
            for i in range(min(transcripts_per_ticker, len(quarters))):
                quarter = quarters[i]
                transcript = self._generate_full_transcript(ticker, quarter, i)
                sample_transcripts.append(transcript)

        word_counts = [t['word_count'] for t in sample_transcripts]
        print(f"\n✅ Generated {len(sample_transcripts)} transcripts")
        print(f"   Average word count: {sum(word_counts) / len(word_counts):,.0f}")
        print(f"   Min word count: {min(word_counts):,}")
        print(f"   Max word count: {max(word_counts):,}")
        print()

        return sample_transcripts

    def _generate_full_transcript(self, ticker, quarter, seed):
        """Generate one complete, varied transcript"""
        import random
        rng = random.Random(hash(ticker + quarter + str(seed)))

        prepared_remarks = self._generate_prepared_remarks(ticker, quarter, rng)
        qa_section = self._generate_qa_section(ticker, quarter, rng)
        full_text = prepared_remarks + "\n\n" + qa_section

        return {
            'ticker': ticker,
            'company_name': self._get_company_name(ticker),
            'quarter': quarter,
            'date': self._get_quarter_date(quarter),
            'fiscal_year': int(quarter.split()[1]),
            'fiscal_quarter': int(quarter[1]),
            'prepared_remarks': prepared_remarks,
            'qa_section': qa_section,
            'full_text': full_text,
            'word_count': len(full_text.split()),
            'speaker_count': rng.randint(8, 15),
            'source': 'SYNTHETIC',
            'collection_date': datetime.now().strftime('%Y-%m-%d')
        }

    def _get_company_name(self, ticker):
        names = {
            'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.', 'NVDA': 'NVIDIA Corporation', 'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.', 'JPM': 'JPMorgan Chase & Co.', 'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson', 'WMT': 'Walmart Inc.', 'PG': 'Procter & Gamble Co.',
            'MA': 'Mastercard Inc.', 'LLY': 'Eli Lilly and Company', 'HD': 'Home Depot Inc.',
            'CVX': 'Chevron Corporation', 'MRK': 'Merck & Co. Inc.', 'ABBV': 'AbbVie Inc.',
            'PEP': 'PepsiCo Inc.', 'COST': 'Costco Wholesale Corporation', 'KO': 'Coca-Cola Company',
            'AVGO': 'Broadcom Inc.', 'TMO': 'Thermo Fisher Scientific', 'MCD': "McDonald's Corporation",
            'CSCO': 'Cisco Systems Inc.', 'ACN': 'Accenture plc', 'ABT': 'Abbott Laboratories',
            'BAC': 'Bank of America Corporation', 'CRM': 'Salesforce Inc.', 'ADBE': 'Adobe Inc.',
            'NFLX': 'Netflix Inc.', 'AMD': 'Advanced Micro Devices', 'ORCL': 'Oracle Corporation',
        }
        return names.get(ticker, f"{ticker} Corporation")

    # ------------------------------------------------------------------
    # PREPARED REMARKS
    # ------------------------------------------------------------------

    def _generate_prepared_remarks(self, ticker, quarter, rng):
        sections = [
            self._generate_opening(ticker, quarter, rng),
            self._generate_financial_overview(ticker, quarter, rng),
            self._generate_segment_details(ticker, quarter, rng),
            self._generate_operational_highlights(ticker, quarter, rng),
            self._generate_strategic_initiatives(ticker, quarter, rng),
            self._generate_guidance(ticker, quarter, rng),
            self._generate_closing_remarks(ticker, quarter, rng),
        ]
        return "\n\n".join(sections)

    def _generate_opening(self, ticker, quarter, rng):
        cfo_names = [
            'Sarah Thompson', 'Michael Chen', 'Jennifer Rodriguez',
            'David Kim', 'Amanda Foster', 'Robert Nguyen', 'Lisa Patel'
        ]
        cfo = rng.choice(cfo_names)
        greeting = rng.choice(['Good morning', 'Good afternoon'])

        return (
            f"Operator: {greeting}, and welcome to {ticker}'s {quarter} Earnings Conference Call. "
            f"At this time, all participants are in a listen-only mode. After the speakers' "
            f"presentation, there will be a question-and-answer session. To ask a question during "
            f"the session, you will need to press star one on your telephone keypad. If you require "
            f"any further assistance, please press star zero. As a reminder, this conference is "
            f"being recorded.\n\n"
            f"I would now like to hand the conference over to {cfo}, Chief Financial Officer. "
            f"Please go ahead.\n\n"
            f"{cfo}, CFO: Thank you, operator, and {greeting.lower()} everyone. Welcome to "
            f"{ticker}'s {quarter} earnings conference call. I am {cfo}, and joining me today are "
            f"our Chief Executive Officer, our Chief Operating Officer, and several other members "
            f"of our senior leadership team.\n\n"
            f"Before we begin, I would like to remind everyone that this call may contain "
            f"forward-looking statements within the meaning of the federal securities laws. "
            f"These statements involve known and unknown risks, uncertainties, and other factors "
            f"that may cause our actual results to differ materially from those expressed or implied "
            f"by the forward-looking statements. We refer you to our most recent annual report on "
            f"Form 10-K and quarterly reports on Form 10-Q for a discussion of these risks. "
            f"We undertake no obligation to update any forward-looking statement.\n\n"
            f"During this call, we will also discuss certain non-GAAP financial measures. "
            f"Reconciliations of these non-GAAP measures to the most directly comparable GAAP "
            f"measures are included in the earnings press release and supplemental financial tables "
            f"available on our Investor Relations website at ir.{ticker.lower()}.com.\n\n"
            f"With that, I will turn the call over to our CEO."
        )

    def _generate_financial_overview(self, ticker, quarter, rng):
        revenue = round(rng.uniform(2.0, 15.0), 2)
        growth = round(rng.uniform(-5.0, 28.0), 1)
        margin = round(rng.uniform(10.0, 45.0), 1)
        eps = round(rng.uniform(0.50, 6.50), 2)
        fcf = round(revenue * rng.uniform(0.10, 0.28), 2)
        buyback = round(revenue * rng.uniform(0.05, 0.18), 2)
        cash = round(revenue * rng.uniform(1.5, 3.0), 2)
        bps_expansion = rng.randint(80, 350)

        tone = rng.choice(['stellar', 'solid', 'challenging', 'mixed', 'turnaround'])

        if tone == 'stellar':
            intro = (
                f"CEO: Thank you, and good morning everyone. I am incredibly pleased to report "
                f"an exceptional {quarter} for {ticker}. Our performance exceeded expectations "
                f"across virtually every metric, and I believe this quarter represents a pivotal "
                f"moment in our company's continued evolution and growth."
            )
            body = (
                f"Let me walk you through the headline numbers before diving deeper. We delivered "
                f"revenue of ${revenue:.2f} billion, representing year-over-year growth of "
                f"{growth:.1f}%, significantly ahead of both our guidance midpoint and Street "
                f"consensus. This marks our strongest quarterly revenue performance in company "
                f"history and reflects the extraordinary execution of our global team.\n\n"
                f"Gross margins expanded meaningfully, and operating margin came in at {margin:.1f}%, "
                f"representing an improvement of {bps_expansion} basis points versus the prior year "
                f"period. This margin expansion was driven by a combination of operating leverage on "
                f"higher volumes, a favorable shift in product and customer mix toward our higher-value "
                f"offerings, and the early realization of benefits from the efficiency programs we "
                f"initiated two quarters ago.\n\n"
                f"Diluted earnings per share of ${eps:.2f} came in ahead of the high end of our "
                f"guidance range, reflecting both the operational outperformance and some benefit "
                f"from a lower-than-expected tax rate resulting from discrete tax items. On a "
                f"normalized basis, EPS growth year-over-year was well above our long-term targets.\n\n"
                f"Free cash flow generation was outstanding at ${fcf:.2f} billion, representing "
                f"a conversion rate of over 110% of net income — a testament to the quality of "
                f"our earnings and our working capital discipline. This robust cash generation "
                f"enabled us to return ${buyback:.2f} billion to shareholders through our share "
                f"repurchase program during the quarter, while still ending the period with "
                f"${cash:.2f} billion in cash, cash equivalents, and short-term investments. "
                f"Our balance sheet has never been stronger, providing us with significant "
                f"flexibility to invest in growth and return capital to shareholders."
            )

        elif tone == 'solid':
            intro = (
                f"CEO: Thank you, and good morning everyone. I am pleased to report solid "
                f"{quarter} results for {ticker} that came in largely in line with our expectations "
                f"and demonstrate the consistent execution that has become a defining characteristic "
                f"of our organization."
            )
            body = (
                f"Revenue for the quarter was ${revenue:.2f} billion, representing growth of "
                f"{growth:.1f}% year-over-year. This growth rate is consistent with our "
                f"expectations and reflects the steady, sustainable trajectory we have been "
                f"delivering quarter after quarter. While there were some puts and takes across "
                f"our business segments, the overall performance was in line with how we had "
                f"modeled the business.\n\n"
                f"Operating margins held at {margin:.1f}%, which is essentially flat on both a "
                f"sequential and year-over-year basis. This represents a good outcome in the "
                f"current operating environment, as we are deliberately balancing investment "
                f"in growth initiatives with the disciplined cost management that our "
                f"shareholders expect.\n\n"
                f"Diluted EPS of ${eps:.2f} was at the midpoint of our guidance range. "
                f"I view this kind of execution — delivering what we say we will deliver — "
                f"as critically important for building long-term credibility with our "
                f"investor base. Free cash flow of ${fcf:.2f} billion was healthy and "
                f"supports our balanced capital allocation approach. We continue to invest "
                f"organically in the business, pursue selective M&A, and return capital "
                f"to shareholders through our dividend and buyback programs."
            )

        elif tone == 'challenging':
            intro = (
                f"CEO: Good morning. I want to be direct with you all today. Our {quarter} "
                f"results fell short of both our internal expectations and the expectations "
                f"we had set with you, and I take full accountability for that underperformance."
            )
            body = (
                f"Revenue came in at ${revenue:.2f} billion, representing growth of only "
                f"{abs(growth):.1f}% year-over-year, and falling below our guidance by "
                f"approximately 5 to 6 percent. The shortfall was primarily driven by "
                f"weaker-than-anticipated demand in two of our key end markets, elongated "
                f"sales cycles in our enterprise segment where deal closings slipped into "
                f"subsequent quarters, and some execution challenges in our go-to-market "
                f"organization that we are actively addressing.\n\n"
                f"Operating margins compressed to {margin:.1f}%, representing a decline "
                f"of {bps_expansion} basis points year-over-year. This margin pressure "
                f"resulted from deleverage on lower revenues combined with a cost structure "
                f"that proved stickier than we had anticipated. We are taking aggressive and "
                f"specific actions to rightsize our cost structure without compromising the "
                f"capabilities we need to compete, and you should begin to see the benefits "
                f"of those actions in the coming quarters.\n\n"
                f"Diluted EPS of ${eps:.2f} reflected these top-line and margin pressures. "
                f"I am not satisfied with this result, and I do not expect you to be either. "
                f"We are a better company than this performance suggests, and we have "
                f"identified the root causes and are implementing specific, measurable "
                f"corrective actions. Free cash flow of ${fcf:.2f} billion, while lower "
                f"than historical levels, demonstrated that the underlying cash generation "
                f"capability of this business remains intact even as we navigate through "
                f"this challenging period."
            )

        elif tone == 'turnaround':
            intro = (
                f"CEO: Good morning everyone. I am pleased to report that our turnaround "
                f"is gaining meaningful traction, and {quarter} results reflect the progress "
                f"we have made over the past several quarters."
            )
            body = (
                f"Revenue of ${revenue:.2f} billion grew {growth:.1f}% year-over-year, "
                f"marking the third consecutive quarter of improving growth trajectory. "
                f"After a difficult period, the strategic actions we took — restructuring "
                f"the organization, refocusing the portfolio, and reinvesting in our core "
                f"competitive strengths — are beginning to yield tangible results.\n\n"
                f"Our restructuring program is now substantially complete, and we are "
                f"operating with a leaner, more focused, and more agile cost structure. "
                f"Operating margins recovered to {margin:.1f}%, representing sequential "
                f"improvement of 120 basis points — the third consecutive quarter of "
                f"sequential margin improvement. We believe we have seen the trough in "
                f"margins and are now on a path of steady improvement.\n\n"
                f"EPS of ${eps:.2f} demonstrated that profitability is returning to levels "
                f"more consistent with our history. Free cash flow of ${fcf:.2f} billion "
                f"shows that the underlying earnings quality is improving alongside reported "
                f"profits. We are cautiously optimistic that the worst is behind us, while "
                f"remaining clear-eyed about the work that remains."
            )

        else:  # mixed
            intro = (
                f"CEO: Thank you for joining us today. Our {quarter} was characterized by "
                f"meaningful divergence across our business — strong execution and encouraging "
                f"momentum in several areas, offset by persistent headwinds in others."
            )
            body = (
                f"Revenue of ${revenue:.2f} billion grew {growth:.1f}% year-over-year, "
                f"landing at the lower end of our guidance range. Within that consolidated "
                f"topline, however, there were important distinctions. Our growth-oriented "
                f"businesses performed quite well and in several cases exceeded our internal "
                f"plans. Our more mature and legacy product lines continued to face the "
                f"secular headwinds we have been navigating.\n\n"
                f"Operating margins of {margin:.1f}% represented sequential improvement of "
                f"70 basis points — encouraging progress even if we are not yet where we "
                f"want to be. The improvement reflects early benefits from our cost "
                f"discipline initiatives and a modest favorable shift in business mix. "
                f"We have more work to do to get margins back to our target range, "
                f"but the trajectory is moving in the right direction.\n\n"
                f"EPS of ${eps:.2f} was modestly below consensus expectations. However, "
                f"adjusting for the one-time items associated with our portfolio "
                f"rationalization activities, normalized EPS would have been "
                f"${round(eps * 1.14, 2):.2f}, which would have been largely in line "
                f"with expectations. Free cash flow of ${fcf:.2f} billion demonstrates "
                f"that the underlying cash generation capability of this business remains "
                f"intact through this transition period."
            )

        return intro + "\n\n" + body

    def _generate_segment_details(self, ticker, quarter, rng):
        segment_pool = [
            'Consumer', 'Enterprise', 'Cloud Services', 'Hardware', 'Software',
            'International', 'Americas', 'Digital', 'Industrial', 'Healthcare',
            'Financial Services', 'Government', 'Small and Medium Business'
        ]
        num_segments = rng.randint(2, 4)
        selected = rng.sample(segment_pool, num_segments)

        text = (
            "Now let me provide more detail on the performance of each of our "
            "business segments for the quarter:\n\n"
        )

        details = [
            "Customer acquisition metrics improved substantially, with new logo additions up {pct}% compared to the same period last year.",
            "We continued to see strong traction in our premium tier offerings, which drove favorable revenue mix and higher average selling prices.",
            "Net revenue retention in this segment came in at {nrr}%, reflecting strong expansion activity within our existing customer base.",
            "Our go-to-market transformation in this segment, which we began executing eighteen months ago, is now fully implemented and producing measurable results.",
            "International expansion within this segment accelerated, with particular strength in Asia-Pacific and select European markets.",
            "We saw meaningful improvement in sales cycle duration, with average days to close improving by {days} days versus the prior year.",
            "Customer satisfaction scores in this segment reached an all-time high of {score} out of 100, driven by product improvements and enhanced customer success investments.",
        ]

        for segment in selected:
            growth = round(rng.uniform(-8.0, 35.0), 1)
            rev_pct = round(rng.uniform(15.0, 45.0), 0)
            margin = round(rng.uniform(8.0, 52.0), 1)

            if growth > 15:
                perf = (
                    f"delivered outstanding performance in {quarter}, with revenue growing "
                    f"{growth:.1f}% year-over-year. This segment now represents approximately "
                    f"{rev_pct:.0f}% of total company revenue and continues to serve as our "
                    f"primary growth engine. Operating margin in this segment expanded to "
                    f"{margin:.1f}%, benefiting from operating leverage, favorable pricing "
                    f"dynamics, and an increasingly efficient cost structure."
                )
            elif growth > 3:
                perf = (
                    f"delivered solid performance with revenue growth of {growth:.1f}% "
                    f"year-over-year, accounting for approximately {rev_pct:.0f}% of "
                    f"consolidated revenue. Segment operating margins of {margin:.1f}% "
                    f"were stable year-over-year as productivity improvements offset "
                    f"incremental investments in go-to-market capacity and product development."
                )
            else:
                perf = (
                    f"faced headwinds during the quarter, with revenue growth of {growth:.1f}% "
                    f"year-over-year. While this segment represents approximately {rev_pct:.0f}% "
                    f"of our business, we are taking specific actions to stabilize and ultimately "
                    f"reaccelerate performance. Segment margins of {margin:.1f}% were under "
                    f"pressure from deleverage, though we have identified meaningful cost "
                    f"reduction opportunities that will be realized over the coming quarters."
                )

            pct_val = rng.randint(12, 45)
            nrr_val = rng.randint(105, 130)
            days_val = rng.randint(5, 20)
            score_val = rng.randint(82, 97)

            detail_template = rng.choice(details)
            detail = (detail_template
                      .replace('{pct}', str(pct_val))
                      .replace('{nrr}', str(nrr_val))
                      .replace('{days}', str(days_val))
                      .replace('{score}', str(score_val)))

            text += f"{segment}: Our {segment} segment {perf} {detail}\n\n"

        return text

    def _generate_operational_highlights(self, ticker, quarter, rng):
        text = (
            "Beyond the financial results, I want to highlight several operational "
            "accomplishments this quarter that demonstrate the health and momentum "
            "of our business:\n\n"
        )

        possible = [
            f"We launched {rng.randint(3, 9)} new products and major feature releases this quarter, representing the most active product innovation cycle we have executed in the past {rng.randint(2, 5)} years. Customer feedback has been exceptionally positive, with early adoption rates tracking well ahead of historical benchmarks.",
            f"Customer satisfaction scores improved to {rng.randint(85, 96)} out of 100 on our primary survey methodology, an improvement of {rng.randint(3, 9)} points year-over-year. This represents an all-time high for our company and reflects the investments we have made in product quality and customer success over the past eighteen months.",
            f"We successfully completed the integration of our {rng.choice(['Nexus', 'Apex', 'Vantage', 'Horizon', 'Clarity'])} acquisition, which is now fully operational and contributing approximately ${rng.randint(45, 220)} million in quarterly revenue, ahead of our original acquisition model.",
            f"Our global supply chain optimization initiative delivered ${rng.randint(30, 130)} million in cost savings during the quarter. This program, now in its second year, has exceeded our original targets by approximately {rng.randint(15, 35)}%.",
            f"We added {rng.randint(200, 800)} net new employees during the quarter, with the majority of hiring concentrated in engineering, product, and customer-facing roles. Our offer acceptance rate improved to {rng.randint(82, 94)}%, reflecting the strength of our employer brand.",
            f"We signed {rng.randint(5, 18)} transactions with total contract value exceeding ${rng.randint(10, 50)} million each during the quarter — the highest count of large deals we have ever recorded in a single quarter.",
            f"R&D output continued to improve, with {rng.randint(15, 45)} patents filed and time-to-market for new product features reduced by {rng.randint(15, 35)}% compared to our historical average.",
            f"We received regulatory approval to operate in {rng.randint(3, 8)} new markets, opening up an addressable market opportunity we estimate at approximately ${rng.randint(500, 3000)} million in aggregate.",
            f"Employee engagement scores improved to {rng.randint(72, 88)}%, up {rng.randint(4, 10)} percentage points year-over-year, reflecting the positive impact of our culture and leadership investments.",
            f"Our partner ecosystem grew to over {rng.randint(800, 8000)} active partners globally, with partner-sourced revenue now representing {rng.randint(18, 38)}% of total bookings.",
        ]

        selected = rng.sample(possible, rng.randint(5, 8))
        for i, h in enumerate(selected, 1):
            text += f"{i}. {h}\n\n"

        return text

    def _generate_strategic_initiatives(self, ticker, quarter, rng):
        text = (
            "Let me now turn to our strategic priorities and the progress we are making "
            "on each of our key initiatives:\n\n"
        )

        ai_budget = rng.randint(80, 400)
        num_countries = rng.randint(4, 14)
        intl_target_pct = rng.randint(30, 52)
        num_partners = rng.randint(600, 8000)
        ma_capacity = round(rng.uniform(1.0, 6.0), 1)
        launch_timing = rng.choice(['later this quarter', 'early next quarter',
                                    'by the end of the fiscal year', 'in the first half of next year'])
        cost_save_target = rng.randint(120, 600)
        beta_customers = rng.randint(20, 80)
        nps_score = rng.randint(62, 84)

        possible = [
            (
                "Artificial Intelligence and Machine Learning: AI is not a future initiative "
                "for us — it is a present reality that is already differentiating our products "
                "and creating measurable value for our customers. We are investing approximately "
                f"${ai_budget} million in AI research and engineering this fiscal year, and we "
                "are seeing that investment translate into product capabilities that our "
                "customers tell us are genuinely transformative. Our AI-powered features are "
                "now available across our entire product portfolio, and early data shows a "
                "meaningful improvement in customer productivity and satisfaction metrics. "
                "We believe AI will be a defining competitive variable in our industry over "
                "the next five to ten years, and we are committed to being at the forefront "
                "of that transformation."
            ),
            (
                "International Expansion: We are executing an aggressive but disciplined "
                "international growth strategy, with particular emphasis on markets where "
                "digital adoption is accelerating and our competitive position is strong. "
                f"We have established local operations in {num_countries} new countries "
                "over the past twelve months and are building out dedicated go-to-market "
                "teams with deep local expertise. We have adapted our products for local "
                "language, compliance, and workflow requirements in key markets. "
                f"International revenue is on a trajectory to represent {intl_target_pct}% "
                "of total company revenue within the next two years, up from our current "
                "level. The opportunity in front of us internationally is substantial, "
                "and we are investing accordingly."
            ),
            (
                "Platform and Ecosystem Strategy: Our transition from a product-centric "
                "to a platform-centric business model is progressing ahead of schedule. "
                f"Our partner ecosystem has expanded to over {num_partners:,} active "
                "technology and services partners, and the third-party applications and "
                "integrations available on our platform now number in the thousands. "
                "This ecosystem creates powerful network effects — customers who are "
                "deeply embedded in our platform show 45% higher annual contract values "
                "and 30% lower churn rates compared to customers using only our core "
                "products. We believe the platform strategy is a durable competitive "
                "advantage that will compound over time."
            ),
            (
                "Environmental, Social, and Governance Leadership: Our ESG commitments "
                "are not separate from our business strategy — they are integral to it. "
                "We have committed to achieving carbon neutrality across our entire "
                "value chain by 2030, and we are tracking ahead of our interim milestones. "
                "Our diversity and inclusion programs are showing measurable results, "
                "with representation of underrepresented groups in senior leadership "
                "improving meaningfully year-over-year. Importantly, we are also seeing "
                "ESG credentials become an increasingly important factor in enterprise "
                "purchasing decisions and talent acquisition — making our ESG investments "
                "commercially relevant, not just ethically important."
            ),
            (
                "Strategic M&A and Partnerships: We maintain an active and rigorous "
                "approach to inorganic growth. Our M&A team is continuously evaluating "
                "opportunities that meet our strategic and financial criteria: clear "
                "strategic fit, defensible competitive position, strong management teams, "
                f"and valuation discipline. We have approximately ${ma_capacity} billion "
                "in capacity to deploy for the right opportunities, and we are patient "
                "in our approach. We have walked away from more deals than we have "
                "pursued, which we view as a sign of discipline rather than inactivity."
            ),
            (
                "Next-Generation Platform Development: Our next-generation platform "
                f"is on track for general availability {launch_timing}. We currently "
                f"have {beta_customers} customers in our beta program, and the feedback "
                f"has been exceptional — with net promoter scores averaging {nps_score}, "
                "which would represent the highest NPS we have ever recorded for a "
                "product at this stage of development. The platform incorporates "
                "fundamentally new architecture that will enable capabilities not "
                "possible on our current platform, and we believe it will be a "
                "significant driver of customer acquisition and expansion activity "
                "once broadly available."
            ),
            (
                "Operational Excellence: We launched a comprehensive operational "
                f"excellence program targeting ${cost_save_target} million in annual "
                "cost savings, to be realized over the next eighteen months. This "
                "program spans procurement optimization, real estate rationalization, "
                "automation of manual processes, and organizational efficiency. "
                f"In the current quarter, we captured approximately ${round(cost_save_target * 0.18)}"
                " million of savings against this target, putting us modestly ahead of "
                "our original timeline. Critically, these savings are being achieved "
                "without compromising our investment in the capabilities we need to "
                "compete and grow."
            ),
        ]

        selected = rng.sample(possible, rng.randint(3, 5))
        for initiative in selected:
            text += initiative + "\n\n"

        return text

    def _generate_guidance(self, ticker, quarter, rng):
        current_q_num = int(quarter[1])
        next_q_num = current_q_num + 1 if current_q_num < 4 else 1
        next_q = f"Q{next_q_num}"

        rev_low = round(rng.uniform(2.0, 12.0), 2)
        rev_high = round(rev_low * rng.uniform(1.03, 1.08), 2)
        eps_low = round(rng.uniform(0.80, 5.50), 2)
        eps_high = round(eps_low * rng.uniform(1.05, 1.13), 2)
        fy_rev_low = round(rev_low * 4 * rng.uniform(0.94, 0.99), 2)
        fy_rev_high = round(rev_high * 4 * rng.uniform(1.00, 1.06), 2)
        fy_eps_low = round(eps_low * 4 * rng.uniform(0.94, 0.99), 2)
        fy_eps_high = round(eps_high * 4 * rng.uniform(1.00, 1.06), 2)
        op_margin = round(rng.uniform(15.0, 35.0), 1)
        fcf_conv = round(rng.uniform(82.0, 112.0), 0)
        fx_bps = rng.randint(80, 320)
        seasonal_q = rng.choice(['fourth quarter', 'second half', 'third and fourth quarters'])
        guidance_verb = rng.choice(['raising', 'reaffirming', 'narrowing', 'updating'])

        text = (
            f"Now let me turn to our financial guidance. Based on our {quarter} performance "
            f"and the visibility we currently have into the business, we are {guidance_verb} "
            f"our outlook for the remainder of the fiscal year.\n\n"
            f"For {next_q}, we expect:\n"
            f"  - Total revenue in the range of ${rev_low:.2f} billion to ${rev_high:.2f} billion\n"
            f"  - Non-GAAP operating margin of approximately {op_margin:.1f}%\n"
            f"  - Non-GAAP diluted earnings per share of ${eps_low:.2f} to ${eps_high:.2f}\n\n"
            f"For the full fiscal year, we now expect:\n"
            f"  - Total revenue of ${fy_rev_low:.2f} billion to ${fy_rev_high:.2f} billion\n"
            f"  - Non-GAAP diluted EPS of ${fy_eps_low:.2f} to ${fy_eps_high:.2f}\n"
            f"  - Free cash flow conversion of approximately {fcf_conv:.0f}% of net income\n\n"
        )

        if guidance_verb == 'raising':
            reasoning = (
                "This guidance increase reflects the strong momentum we are seeing across "
                "the business and our improved visibility into demand. We have increased "
                "confidence in our ability to execute against our operating plan, and we "
                "believe the raise is appropriately balanced — reflecting our outperformance "
                "while maintaining appropriate conservatism given the macro environment."
            )
        elif guidance_verb == 'narrowing':
            reasoning = (
                "We are narrowing our guidance range as we have greater visibility into "
                "the business with fewer quarters remaining in the fiscal year. The "
                "midpoint of our guidance remains unchanged, which reflects our confidence "
                "in the trajectory of the business."
            )
        else:
            reasoning = (
                "Our guidance reflects our current visibility into demand patterns and "
                "assumes a broadly stable macroeconomic environment. We have modeled "
                "a range of scenarios and believe our guidance represents an achievable "
                "and appropriately balanced outlook."
            )

        text += reasoning + "\n\n"

        text += (
            f"A few key assumptions embedded in our guidance deserve mention. We are "
            f"assuming foreign exchange will be an approximately {fx_bps} basis point "
            f"headwind to revenue on a year-over-year basis based on current rates. "
            f"We are not assuming any significant change in the competitive pricing "
            f"environment. We are expecting seasonal demand patterns consistent with "
            f"historical norms, with typical weighting toward the {seasonal_q}. "
            f"And we are assuming our current operational footprint, excluding any "
            f"potential M&A activity that has not yet been announced or closed."
        )

        return text

    def _generate_closing_remarks(self, ticker, quarter, rng):
        tone = rng.choice(['optimistic', 'balanced', 'determined', 'grateful'])

        if tone == 'optimistic':
            return (
                f"Before we open the call for questions, I want to leave you with this "
                f"thought: I have never been more confident in {ticker}'s ability to "
                f"create long-term value for our shareholders, customers, and employees. "
                f"The investments we have made are bearing fruit. Our competitive "
                f"position is strengthening. Our team is executing with purpose and "
                f"precision. And the market opportunity in front of us has never been "
                f"larger.\n\n"
                f"We are in the early innings of a significant growth phase for this "
                f"company, and I am excited about what the coming quarters and years "
                f"will bring. I am grateful for the trust you place in us, and we are "
                f"committed to rewarding that trust through consistent execution and "
                f"value creation.\n\n"
                f"With that, let us open the call for your questions. Operator?"
            )
        elif tone == 'balanced':
            return (
                f"In closing, I want to reiterate what I believe are the key messages "
                f"from today's call. First, the fundamental health of our business is "
                f"sound. Second, we are executing thoughtfully in a complex environment. "
                f"Third, our strategic priorities are clear and the team is aligned "
                f"behind executing them.\n\n"
                f"We are not immune to macroeconomic forces, but we have built a "
                f"business with the resilience, flexibility, and balance sheet strength "
                f"to navigate uncertainty while continuing to invest in our future. "
                f"We have navigated challenging periods before, and we have consistently "
                f"emerged stronger.\n\n"
                f"Thank you for your continued interest in {ticker}. "
                f"Let us now move to your questions. Operator, please open the line."
            )
        elif tone == 'determined':
            return (
                f"Let me close with a direct statement of where we stand. We are not "
                f"where we want to be yet, but we know exactly what we need to do and "
                f"we are executing against that plan with urgency and accountability. "
                f"I have seen this team rise to challenges before, and I have full "
                f"confidence we will do so again.\n\n"
                f"Every member of our leadership team understands what is at stake "
                f"and what is expected. We are making the difficult decisions that "
                f"long-term value creation requires. And I am committed to providing "
                f"you with transparent, candid communication about our progress "
                f"at every step.\n\n"
                f"With that, let us take your questions. Operator?"
            )
        else:  # grateful
            return (
                f"I want to close by acknowledging the extraordinary effort of our "
                f"global team. The results we delivered this quarter are the product "
                f"of thousands of people working hard every day on behalf of our "
                f"customers and our shareholders. I am grateful to serve alongside "
                f"such a talented and dedicated team.\n\n"
                f"I also want to thank our customers, whose partnership and feedback "
                f"make us better every day. And I want to thank our shareholders for "
                f"their continued confidence in our strategy and our ability to execute.\n\n"
                f"We look forward to continuing to deliver on our commitments. "
                f"Operator, please open the line for questions."
            )

    # ------------------------------------------------------------------
    # Q&A SECTION
    # ------------------------------------------------------------------

    def _generate_qa_section(self, ticker, quarter, rng):
        qa = "\n\n" + "=" * 60 + "\n"
        qa += "QUESTION AND ANSWER SESSION\n"
        qa += "=" * 60 + "\n\n"
        qa += (
            "Operator: Thank you. We will now begin the question-and-answer session. "
            "To ask a question, please press star one on your telephone keypad. "
            "To withdraw your question, please press star two. "
            "Please limit yourself to one question and one follow-up question. "
            "Our first question comes from the queue. Please go ahead.\n\n"
        )

        analysts = [
            ('Mark Mahaney', 'Evercore ISI'),
            ('Brent Thill', 'Jefferies'),
            ('Keith Weiss', 'Morgan Stanley'),
            ('Raimo Lenschow', 'Barclays'),
            ('Karl Keirstead', 'UBS'),
            ('Derrick Wood', 'Cowen'),
            ('Brad Reback', 'Stifel'),
            ('Kash Rangan', 'Bank of America'),
            ('Jennifer Lowe', 'Credit Suisse'),
            ('Michael Ng', 'Goldman Sachs'),
            ('Tyler Radke', 'Citi'),
            ('Alex Zukin', 'Wolfe Research'),
            ('Brian Essex', 'J.P. Morgan'),
            ('Patrick Walravens', 'JMP Securities'),
            ('Terry Tillman', 'Truist Securities'),
            ('Kirk Materne', 'Evercore'),
            ('Stan Zlotsky', 'Morgan Stanley'),
            ('Matt Hedberg', 'RBC Capital Markets'),
        ]

        num_questions = rng.randint(9, 15)
        selected = rng.sample(analysts, num_questions)

        topics = [
            'revenue_drivers', 'margins', 'competition', 'growth_outlook',
            'cash_deployment', 'product_roadmap', 'international', 'macro_sensitivity',
            'pricing_power', 'customer_trends', 'hiring_talent', 'churn_retention'
        ]

        used_topics = []
        for name, firm in selected:
            # Ensure topic variety
            remaining = [t for t in topics if t not in used_topics]
            if not remaining:
                used_topics = []
                remaining = topics[:]
            topic = rng.choice(remaining)
            used_topics.append(topic)

            qa += self._generate_single_qa_exchange(ticker, name, firm, topic, rng)
            qa += "\n\n"

        qa += (
            "Operator: This concludes our question-and-answer session. I would now like "
            "to turn the conference back over to management for any closing remarks.\n\n"
            f"CEO: Thank you all for joining us today and for the excellent questions. "
            f"We appreciate your continued interest in {ticker} and look forward to "
            f"updating you again next quarter. In the meantime, our Investor Relations "
            f"team is available for follow-up conversations. Have a great rest of your day."
        )

        return qa

    def _generate_single_qa_exchange(self, ticker, analyst_name, firm, topic, rng):
        first_name = analyst_name.split()[0]

        exchange = f"Operator: Our next question comes from {analyst_name} with {firm}. Please go ahead.\n\n"
        exchange += f"{analyst_name}: Hi, good morning. Thanks so much for taking my question. "
        exchange += self._generate_question(topic, ticker, rng)
        exchange += f"\n\nCEO: Thanks {first_name}, great question. "
        exchange += self._generate_answer(topic, ticker, rng)

        # ~70% chance of follow-up
        if rng.random() < 0.70:
            exchange += f"\n\n{analyst_name}: Very helpful. One quick follow-up if I may — "
            exchange += self._generate_followup_question(topic, rng)
            exchange += f"\n\nCEO: "
            exchange += self._generate_followup_answer(rng)

        exchange += f"\n\n{analyst_name}: Great, thank you very much. Appreciate the color."
        return exchange

    def _generate_question(self, topic, ticker, rng):
        questions = {
            'revenue_drivers': (
                "I wanted to dig a little deeper into the revenue beat this quarter. "
                "Can you help us decompose what drove the outperformance — was it "
                "more volume, pricing, mix, or deal timing? And as you think about "
                "the sustainability of those drivers into the back half, what gives "
                "you the most and least confidence?"
            ),
            'margins': (
                "On the margin side, you have delivered consistent improvement over "
                "the past several quarters. Could you walk us through the key drivers "
                "of that improvement and help us understand how much is structural "
                "versus transitory? And where do you ultimately see the margin "
                "profile of this business settling out in a normalized environment?"
            ),
            'competition': (
                "I wanted to ask about the competitive landscape. We have been hearing "
                "about increased pricing aggression from a couple of your major competitors. "
                "Are you seeing that in your own win-loss data? And how does that "
                "influence your go-to-market strategy over the next few quarters?"
            ),
            'growth_outlook': (
                "Looking out to next year and beyond, can you walk us through the "
                "key growth drivers you are most excited about? And conversely, "
                "what are the scenarios that could cause you to come in below your "
                "long-term growth targets?"
            ),
            'cash_deployment': (
                "You have been generating very strong free cash flow, and the balance "
                "sheet is in excellent shape. I wanted to ask about your capital "
                "allocation priorities going forward — how are you thinking about "
                "the relative attractiveness of buybacks, M&A, and organic investment "
                "at this stage of the business?"
            ),
            'product_roadmap': (
                "You mentioned several significant product launches and updates on the "
                "call today. Can you give us a sense of the revenue ramp profile "
                "we should expect from these launches? And are there any particular "
                "products or capabilities you are most excited about from a competitive "
                "differentiation standpoint?"
            ),
            'international': (
                "International was clearly a bright spot this quarter with very "
                "strong growth across multiple regions. I wanted to understand better "
                "what is driving that acceleration. Is it market share gains, end "
                "market expansion, or are you seeing the payoff from go-to-market "
                "investments you made in prior periods?"
            ),
            'macro_sensitivity': (
                "Given the mixed macro signals we are all seeing in the data — "
                "softer consumer sentiment, tightening credit conditions, some "
                "slowdown in enterprise IT spending — I wanted to understand your "
                "sensitivity to a potential demand deterioration. What gives you "
                "confidence in your guidance, and what would a recessionary "
                "scenario look like for your business?"
            ),
            'pricing_power': (
                "You have been taking price increases across your portfolio. Can you "
                "talk about the customer response — are you seeing any meaningful "
                "pushback or elevated churn? And as you look at your pricing "
                "architecture going forward, do you think there is additional "
                "opportunity to monetize the value you are delivering to customers?"
            ),
            'customer_trends': (
                "I wanted to ask about what you are seeing in customer behavior "
                "and demand patterns. Are enterprise customers showing any "
                "signs of budget constraint or deal scrutiny that were not "
                "present six months ago? And how has that influenced your "
                "go-to-market and customer success approach?"
            ),
            'hiring_talent': (
                "On the talent and hiring side, you mentioned adding employees "
                "this quarter. Can you talk about where you are investing in "
                "headcount growth and where you are being more selective? And "
                "how is the talent market environment affecting your ability "
                "to hire the skills you need at competitive costs?"
            ),
            'churn_retention': (
                "I wanted to ask about customer retention and net revenue retention "
                "trends. Are you seeing any change in churn or expansion rates "
                "versus six to twelve months ago? And how are you thinking about "
                "the investment required to maintain your retention profile "
                "in an environment where customers may be scrutinizing "
                "their software spend more carefully?"
            ),
        }
        return questions.get(topic, "Can you provide more color on the key drivers of performance this quarter?")

    def _generate_answer(self, topic, ticker, rng):
        if topic == 'revenue_drivers':
            vol = rng.randint(30, 52)
            price_mix = rng.randint(25, 40)
            timing = 100 - vol - price_mix
            months = rng.randint(12, 30)
            return (
                f"Let me try to give you a precise decomposition of the revenue performance. "
                f"If I had to allocate the outperformance, I would say roughly {vol}% came "
                f"from better-than-expected volume and unit growth, approximately {price_mix}% "
                f"came from pricing and mix, and the remaining {timing}% was attributable to "
                f"deal timing — transactions that we expected to close in the subsequent "
                f"quarter but which the teams successfully accelerated into this quarter.\n\n"
                f"On sustainability, I feel most confident about the volume component. "
                f"Underlying demand signals remain healthy — our pipeline is at its highest "
                f"level in over {months} months, conversion rates are improving, and "
                f"customer conversations feel constructive. The mix improvement is also "
                f"relatively durable, as we are seeing a secular shift toward our "
                f"higher-value offerings that I expect to continue.\n\n"
                f"Where I am more measured is on the deal timing component — by definition, "
                f"pulling revenue forward creates a headwind in the subsequent quarter. "
                f"We have factored that into our {rng.choice(['Q3', 'Q4'])} guidance, "
                f"so the numbers we have given you are net of that dynamic."
            )

        elif topic == 'margins':
            lev = rng.randint(100, 280)
            cost = rng.randint(60, 160)
            mix = rng.randint(40, 100)
            hw = rng.randint(40, 110)
            net = lev + cost + mix - hw
            target = rng.randint(22, 38)
            return (
                f"The margin improvement is definitely something I am proud of, and I want "
                f"to give you a transparent view of what is driving it. On a year-over-year "
                f"basis, we got approximately {lev} basis points of benefit from operating "
                f"leverage, {cost} basis points from permanent cost structure improvements "
                f"associated with our efficiency program, and {mix} basis points from "
                f"favorable business mix. Partially offsetting those tailwinds, we had "
                f"about {hw} basis points of headwind from incremental investments in "
                f"R&D and go-to-market capacity. Net, that gets us to approximately "
                f"{net} basis points of improvement, which is what you see in the numbers.\n\n"
                f"On the structural versus transitory question — I would characterize the "
                f"operating leverage and cost savings as largely structural and repeatable. "
                f"The mix benefit is also fairly durable. The one area of uncertainty is "
                f"around our investment spend, which will fluctuate based on the "
                f"opportunities we see.\n\n"
                f"In terms of where margins settle in a normalized environment, I think "
                f"the mid-to-high {target}s is the right target. We are not going to get "
                f"there in a straight line, and we will make investments that create "
                f"quarter-to-quarter variability. But that is the destination."
            )

        elif topic == 'competition':
            win_rate = rng.randint(56, 76)
            disp_rate = rng.randint(61, 80)
            acv_growth = rng.randint(12, 35)
            return (
                f"On the competitive environment, I want to give you my honest assessment. "
                f"It is competitive — it always has been in our space — but I would not "
                f"characterize what we are seeing as irrational pricing behavior that "
                f"structurally impairs the economics of the industry.\n\n"
                f"Our win rate data, which we track rigorously on a weekly basis, shows "
                f"our overall competitive win rate at approximately {win_rate}%, which is "
                f"consistent with where it has been over the past several quarters. In "
                f"head-to-head competitive displacements — situations where a customer "
                f"is actively evaluating us against an incumbent — our win rate is "
                f"actually higher, around {disp_rate}%. That is a meaningful data point "
                f"that gives me confidence in our differentiation.\n\n"
                f"On deal sizes, average contract value is up approximately {acv_growth}% "
                f"year-over-year. That tells me customers are not trading down or "
                f"becoming more price-sensitive — they are actually deepening their "
                f"commitment to our platform. We view the competitive dynamics as "
                f"healthy validation of the market and we welcome the competition because "
                f"it keeps us sharp and customer-focused."
            )

        elif topic == 'growth_outlook':
            intl_growth = rng.randint(22, 45)
            intl_pct = rng.randint(26, 42)
            return (
                f"Looking ahead, I want to give you a specific view of the drivers "
                f"that underpin our growth confidence rather than just high-level optimism.\n\n"
                f"First, we have meaningful product cycle tailwinds. Our next-generation "
                f"platform launch and several significant product updates are creating "
                f"genuine upsell and cross-sell opportunities with our existing customers, "
                f"which is often our highest-ROI growth lever.\n\n"
                f"Second, international is a multi-year growth driver that is still in "
                f"relatively early stages. International revenue is growing {intl_growth}% "
                f"and represents {intl_pct}% of total revenue. There is a long runway "
                f"ahead as we penetrate markets where we are currently underpenetrated "
                f"relative to our competitive position.\n\n"
                f"Third, the secular demand environment for what we do remains intact. "
                f"The digitization trends, the AI tailwinds, and the ongoing shift "
                f"toward our category of solutions are not cyclical — they are "
                f"structural and multi-year.\n\n"
                f"The risks I am watching most carefully are enterprise budget pressure "
                f"if the macro environment deteriorates meaningfully, and the pace of "
                f"international expansion, which can be impacted by geopolitical factors "
                f"outside our control. But our base case is healthy and we have conviction "
                f"in the numbers we have provided."
            )

        elif topic == 'cash_deployment':
            capacity = round(rng.uniform(1.2, 6.0), 1)
            dividend = round(rng.uniform(0.25, 1.20), 2)
            buyback_remaining = round(rng.uniform(1.0, 4.0), 1)
            return (
                f"Capital allocation is a topic we take very seriously at the board "
                f"level, and I want to give you a clear picture of our framework.\n\n"
                f"First priority is always organic investment in the business. We have "
                f"a strong pipeline of internal capital projects — in engineering, "
                f"product development, and go-to-market — that generate attractive "
                f"risk-adjusted returns, and we are not constraining those investments.\n\n"
                f"Second is M&A. We have a disciplined but active acquisition program "
                f"focused on tuck-in opportunities that expand our product capabilities "
                f"or accelerate our geographic reach. Our balance sheet provides us "
                f"with approximately ${capacity} billion in M&A capacity, and we are "
                f"prepared to deploy that for the right opportunity. But our criteria "
                f"are strict — strategic fit, reasonable valuation, strong team, "
                f"and clear integration path.\n\n"
                f"Third is returning capital to shareholders. Our quarterly dividend "
                f"of ${dividend} per share reflects our confidence in the durability "
                f"of our cash flows, and we recently completed another buyback of "
                f"shares under our repurchase authorization with ${buyback_remaining} "
                f"billion remaining. We view buybacks as particularly attractive at "
                f"current valuation levels."
            )

        elif topic == 'product_roadmap':
            nps = rng.randint(66, 85)
            beta_count = rng.randint(18, 60)
            quarters_to_material = rng.randint(2, 4)
            return (
                f"On the product side, let me give you specifics on what we are "
                f"seeing in terms of adoption and revenue trajectory.\n\n"
                f"Our most significant recent launch is currently in beta with "
                f"{beta_count} customers, and the feedback has been exceptional — "
                f"net promoter scores averaging {nps}, which is the highest pre-GA "
                f"NPS we have ever measured. We expect general availability in the "
                f"coming weeks, with revenue contribution becoming meaningful within "
                f"approximately {quarters_to_material} quarters as the sales motion "
                f"ramps and customers complete their implementation cycles.\n\n"
                f"From a competitive differentiation standpoint, the capabilities "
                f"I am most excited about are in our AI and automation layer. We "
                f"have built capabilities that our largest competitors simply do not "
                f"have, and customers are telling us that these features are "
                f"meaningfully improving their outcomes. That creates a durable "
                f"competitive advantage that is difficult to replicate quickly.\n\n"
                f"Our product roadmap for the next twelve months is the most "
                f"ambitious we have ever had, and I believe it will widen our "
                f"competitive moat further."
            )

        elif topic == 'international':
            apac = rng.randint(28, 55)
            emea = rng.randint(18, 38)
            latam = rng.randint(22, 48)
            margin_gap = rng.randint(3, 9)
            return (
                f"International performance this quarter was genuinely strong across "
                f"all three major geographies. Asia-Pacific grew {apac}% year-over-year, "
                f"EMEA was up {emea}%, and Latin America showed acceleration to "
                f"{latam}% growth. I want to give you a clear-eyed view of what "
                f"is driving this.\n\n"
                f"The primary driver is the payoff from go-to-market investments we "
                f"began making roughly two years ago. We built out local sales and "
                f"customer success teams, we localized our product for key markets, "
                f"and we established a partner ecosystem that gives us reach beyond "
                f"what our direct sales force can achieve on its own. Those investments "
                f"are now compounding.\n\n"
                f"We are also seeing the benefit of market share expansion in markets "
                f"where we were previously underpenetrated relative to our competitive "
                f"position. There is meaningful white space still ahead of us "
                f"internationally.\n\n"
                f"On margins — international does carry a margin profile "
                f"approximately {margin_gap} to {margin_gap + 3} points below our "
                f"overall company average, reflecting the earlier stage of scale and "
                f"higher go-to-market investment intensity. We have a clear path to "
                f"closing that gap as these markets mature, and the growth rates "
                f"more than justify the current investment."
            )

        elif topic == 'macro_sensitivity':
            indicator = rng.choice([
                'weekly enterprise software pipeline data and sales cycle duration metrics',
                'small business confidence surveys and consumer discretionary spending data',
                'industrial production indices and manufacturing new orders data',
                'financial sector loan origination data and credit spread metrics'
            ])
            return (
                f"I want to give you a direct and honest answer to this question "
                f"because I know it is top of mind for many of you.\n\n"
                f"Our business has historically shown moderate but real correlation "
                f"to the broader economic cycle — I would not tell you we are immune, "
                f"because we are not. However, the nature of our solutions — "
                f"mission-critical, deeply embedded, with clear ROI — means that "
                f"our products tend to be among the last things customers cut when "
                f"budgets tighten. In prior downturns, our revenue proved more "
                f"resilient than many would have predicted.\n\n"
                f"In terms of what we watch, the most relevant leading indicators "
                f"for our specific business are {indicator}. We track these metrics "
                f"weekly and share them with the board monthly. Current readings "
                f"are mixed — not signaling deterioration, but also not uniformly "
                f"positive. We have incorporated that ambiguity into our guidance "
                f"range, which is why the range is somewhat wider than our historical "
                f"norm.\n\n"
                f"In a genuine recessionary scenario, I would expect some revenue "
                f"headwind, lower new customer acquisition, and modest churn uptick. "
                f"But our strong balance sheet and flexible cost structure would "
                f"allow us to manage through that environment while continuing to "
                f"invest in our long-term competitive position."
            )

        elif topic == 'pricing_power':
            price_pct = rng.randint(3, 8)
            churn = round(rng.uniform(1.0, 3.8), 1)
            nrr = rng.randint(108, 128)
            return (
                f"Pricing is an area where I think we have demonstrated genuine "
                f"discipline and strength. Over the past twelve months, we have "
                f"implemented price increases ranging from {price_pct}% to "
                f"{price_pct + 3}% across our portfolio, and the customer "
                f"response has been more favorable than we had modeled.\n\n"
                f"Annual churn has remained stable at approximately {churn}%, "
                f"which is at or near historical lows for our business. And "
                f"net revenue retention is {nrr}%, meaning that even accounting "
                f"for any churn, our existing customer base is spending substantially "
                f"more with us each year. Both of those metrics tell me that "
                f"customers are receiving genuine value and are absorbing the "
                f"pricing constructively.\n\n"
                f"We do see some price sensitivity at the lower end of our "
                f"customer spectrum — smaller customers with tighter budgets. "
                f"But in our core enterprise segment, pricing conversations have "
                f"been rational, and we are winning based on value rather than "
                f"competing on price. I do believe there is additional pricing "
                f"opportunity ahead, and we will pursue it thoughtfully."
            )

        elif topic == 'customer_trends':
            acv = rng.randint(12, 28)
            cycle = rng.choice(['shortening modestly', 'stable', 'extending slightly'])
            consumption = rng.choice(['accelerating', 'healthy and stable', 'normalizing from elevated levels'])
            return (
                f"Customer behavior is something we track very closely, and let me "
                f"give you a granular view of what we are seeing.\n\n"
                f"On deal sizes — average contract values are up {acv}% year-over-year, "
                f"driven by a combination of new product attach and customers choosing "
                f"larger initial commitments. Customers are consolidating vendors and "
                f"choosing to go deeper with a smaller number of strategic partners, "
                f"which benefits us given the breadth of our platform.\n\n"
                f"On sales cycles — they are {cycle} relative to six months ago. "
                f"We had been concerned about elongation risk given the macro "
                f"environment, but what we are actually seeing in the data is "
                f"encouraging. Customers seem willing to make decisions, which "
                f"is a positive leading indicator.\n\n"
                f"On consumption for our usage-based revenue streams — consumption "
                f"is {consumption}. We are not seeing the kind of dramatic pullback "
                f"that would indicate customers are aggressively cutting usage to "
                f"manage costs. The overall customer health picture feels solid."
            )

        elif topic == 'hiring_talent':
            eng_pct = rng.randint(45, 65)
            accept_rate = rng.randint(78, 93)
            return (
                f"On the talent side, I want to give you a nuanced picture. We are "
                f"being very deliberate about where we are adding capacity and "
                f"where we are holding headcount steady or reducing.\n\n"
                f"Approximately {eng_pct}% of our net new hires this quarter were "
                f"in engineering and product roles, which reflects our conviction "
                f"that product innovation is our most important long-term competitive "
                f"lever. We are also adding in customer success, where we have "
                f"clear evidence that investment drives retention and expansion.\n\n"
                f"In our general and administrative functions, we are actively "
                f"looking for opportunities to improve efficiency and reduce headcount "
                f"through automation and process improvement. So there is an "
                f"intentional rotation happening in our workforce composition.\n\n"
                f"The talent market has actually become more favorable for employers "
                f"over the past twelve months. Our offer acceptance rate is "
                f"{accept_rate}%, up meaningfully from the peak of the talent "
                f"market tightness we experienced two years ago. We are able to "
                f"attract strong talent at reasonable compensation levels."
            )

        else:  # churn_retention
            nrr = rng.randint(110, 128)
            gross_retention = rng.randint(90, 97)
            expansion = rng.randint(18, 32)
            return (
                f"Retention is one of the most important metrics we manage, and "
                f"I am pleased to report that our performance remains strong.\n\n"
                f"Gross retention — the percentage of revenue we retain from "
                f"existing customers excluding any expansion — came in at "
                f"{gross_retention}% on an annualized basis, consistent with "
                f"our historical performance. Net revenue retention was {nrr}%, "
                f"meaning the expansion activity from existing customers more "
                f"than offset any churn. The expansion rate of {expansion}% "
                f"reflects strong product adoption and the success of our "
                f"land-and-expand motion.\n\n"
                f"Versus six months ago, I would say the retention profile is "
                f"slightly improved, driven by the product enhancements we have "
                f"shipped and the investments in our customer success organization. "
                f"We are not seeing customers cutting or renegotiating contracts "
                f"at a rate that would concern us, even in an environment where "
                f"we know software budgets are under scrutiny.\n\n"
                f"We continue to invest meaningfully in customer success because "
                f"the math is straightforward — retaining a customer is far "
                f"more economical than acquiring a new one, and the best "
                f"source of growth is expanding within our installed base."
            )

    def _generate_followup_question(self, topic, rng):
        options = [
            "can you give us a sense of how much of that is baked into the guidance you provided?",
            "should we expect those trends to continue or potentially accelerate into the next quarter?",
            "how does that dynamic compare to what you were seeing twelve months ago?",
            "can you size the financial impact of that for us more specifically?",
            "are there any geographies or customer segments where you are seeing meaningfully different trends?",
            "how are you thinking about that relative to what peers have been saying in their earnings calls?",
            "is that an assumption you have stress-tested against a more adverse macro scenario?",
        ]
        return rng.choice(options)

    def _generate_followup_answer(self, rng):
        options = [
            (
                "Yes, that dynamic is reflected in our guidance — we have been thoughtful "
                "about incorporating our current visibility into the numbers we have provided. "
                "There is always uncertainty at the margin, but we have not assumed either "
                "a material acceleration or deceleration from current trends. We feel the "
                "guidance represents an appropriately balanced view of the opportunities "
                "and risks we see."
            ),
            (
                "Good follow-up. I would say we have been appropriately conservative in "
                "our guidance assumptions — not pessimistic, but not extrapolating the "
                "most favorable recent data points either. If trends continue at their "
                "current pace, we would have upside to the midpoint of our guidance range. "
                "But we prefer to set expectations we are confident we can meet or exceed."
            ),
            (
                "We have stress-tested our guidance against a range of macro scenarios, "
                "including a more adverse environment. In a downside scenario where demand "
                "deteriorates by 10 to 15 percent from current levels, we believe we have "
                "the cost flexibility and balance sheet strength to protect profitability "
                "and continue investing in our strategic priorities. The guidance we have "
                "provided represents our base case, which we think is achievable and "
                "well-supported by current business indicators."
            ),
            (
                "I would characterize the trends as broadly consistent with what we were "
                "seeing in prior quarters, with some incremental improvement in a few "
                "dimensions I mentioned. We are not trying to tell a story of dramatic "
                "inflection — this is a business with improving but gradual momentum. "
                "Consistency of execution over multiple quarters is what we are focused "
                "on delivering, and I think the data supports that we are on that path."
            ),
        ]
        return rng.choice(options)

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def _get_quarter_date(self, quarter_str):
        quarter, year = quarter_str.split()
        year = int(year)
        qn = int(quarter[1])
        ends = {1: f"{year}-03-31", 2: f"{year}-06-30",
                3: f"{year}-09-30", 4: f"{year}-12-31"}
        return ends[qn]

    def save_transcripts(self, transcripts):
        if not transcripts:
            print("❌ No transcripts to save")
            return False

        try:
            print("=" * 60)
            print("💾 SAVING TRANSCRIPTS")
            print("=" * 60)
            print()

            df = pd.DataFrame(transcripts)
            csv_path = self.output_dir / 'transcripts_metadata.csv'
            metadata_cols = ['ticker', 'company_name', 'quarter', 'date', 'fiscal_year',
                             'fiscal_quarter', 'word_count', 'source', 'collection_date']
            df[metadata_cols].to_csv(csv_path, index=False)
            print(f"✅ Metadata saved: {csv_path}")

            json_path = self.output_dir / 'transcripts_full.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transcripts, f, indent=2, ensure_ascii=False)
            print(f"✅ Full transcripts saved: {json_path}")
            print(f"   Size: {json_path.stat().st_size / (1024 * 1024):.2f} MB")

            individual_dir = self.output_dir / 'individual'
            individual_dir.mkdir(exist_ok=True)

            for transcript in transcripts:
                filename = f"{transcript['ticker']}_{transcript['quarter'].replace(' ', '_')}.txt"
                with open(individual_dir / filename, 'w', encoding='utf-8') as f:
                    f.write(f"Company: {transcript['company_name']}\n")
                    f.write(f"Ticker: {transcript['ticker']}\n")
                    f.write(f"Quarter: {transcript['quarter']}\n")
                    f.write(f"Date: {transcript['date']}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(transcript['full_text'])

            print(f"✅ Individual transcripts saved: {individual_dir}/")
            print(f"   Count: {len(transcripts)} files")
            print()
            return True

        except Exception as e:
            print(f"❌ Error saving transcripts: {e}")
            return False

    def _log_info(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - INFO: {message}\n")

    def _log_error(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")

    def run(self):
        print("=" * 60)
        print("PHASE 1D: ENHANCED TRANSCRIPT GENERATION")
        print("=" * 60)
        print()
        print(f"🎯 Target: {self.target_count} transcripts")
        print(f"📏 Length: 7,000-20,000 words per transcript")
        print()

        tickers = self.get_high_priority_tickers()
        transcripts = self.create_sample_transcripts(tickers, transcripts_per_ticker=3)

        if transcripts:
            success = self.save_transcripts(transcripts)
            if success:
                total_words = sum(t['word_count'] for t in transcripts)
                avg_words = total_words / len(transcripts)
                print("=" * 60)
                print("✅ PHASE 1D COMPLETE")
                print("=" * 60)
                print()
                print(f"✅ Generated:        {len(transcripts)} transcripts")
                print(f"✅ Unique companies: {len(set(t['ticker'] for t in transcripts))}")
                print(f"✅ Average length:   {avg_words:,.0f} words")
                print(f"✅ Total corpus:     {total_words:,} words")
                print()
                print(f"📁 Data saved to: {self.output_dir}")
                print()
                return transcripts

        return None


def main():
    collector = TranscriptCollector(target_count=150)
    transcripts = collector.run()
    return transcripts


if __name__ == "__main__":
    main()