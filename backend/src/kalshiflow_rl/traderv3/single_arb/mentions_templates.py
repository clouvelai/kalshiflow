"""
Event structure templates for blind LLM roleplay simulation.

Each domain (sports, corporate, politics) has different broadcast/transcript structures.
Templates capture:
- Segment breakdown (pre-game, quarters, halftime, etc.)
- Content types (scripted vs impromptu)
- Speaker roles (play-by-play, analyst, sideline)
- Impromptu ratios (where rare mentions are most likely)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.mentions_templates")


@dataclass
class TranscriptSegment:
    """A single segment of a broadcast/transcript."""

    name: str  # "pre_game", "q1", "halftime", etc.
    duration_minutes: int
    content_type: str  # "scripted", "play_by_play", "analysis", "impromptu"
    speakers: List[str]  # ["play_by_play", "color_analyst", "sideline"]
    impromptu_ratio: float  # 0.0-1.0, fraction of unscripted content
    description: str = ""  # Human-readable description for prompts


@dataclass
class DomainTemplate:
    """Event structure template for a specific domain/event type."""

    domain: str  # "sports", "corporate", "politics", "entertainment"
    event_type: str  # "nfl_broadcast", "earnings_call", "speech"
    segments: List[TranscriptSegment]
    total_duration_minutes: int
    speaker_roles: Dict[str, str] = field(default_factory=dict)  # role -> description
    key_moments_prompt: str = ""  # What key moments to generate
    style_notes: str = ""  # Broadcast style guidance

    def get_simulation_prompt(
        self,
        event_title: str,
        participants: List[str],
        announcers: List[str],
        venue: str,
        storylines: List[str],
        special_context: Optional[str] = None,
    ) -> str:
        """Generate the blind roleplay prompt for this template.

        CRITICAL: This prompt must NOT include any mention terms.
        The simulation must be blind to what we're looking for.
        """
        # Build speaker assignments
        speaker_intro = self._format_speaker_intro(announcers)

        # Build storylines (must be pre-filtered to exclude mention terms)
        storylines_text = "\n".join(f"- {s}" for s in storylines[:5]) if storylines else "- No specific storylines"

        # Build participants context
        participants_text = ", ".join(participants) if participants else "TBD"

        # Build key moments section from template
        key_moments = self.key_moments_prompt or self._default_key_moments()

        prompt = f"""You are simulating a realistic {self.event_type} broadcast/transcript.

EVENT: {event_title}
VENUE: {venue}
PARTICIPANTS: {participants_text}

{speaker_intro}

CURRENT CONTEXT/STORYLINES:
{storylines_text}

{f"SPECIAL CONTEXT: {special_context}" if special_context else ""}

STYLE NOTES:
{self.style_notes or self._default_style_notes()}

YOUR TASK:
Generate a realistic transcript covering these key moments:
{key_moments}

IMPORTANT RULES:
1. Write naturally - how a real {self.event_type} would sound
2. Include the natural cadence, reactions, and banter of professional broadcasters
3. Do NOT artificially insert any specific words - speak as you naturally would
4. Include both scripted segments (stats, promos) and impromptu reactions
5. Each segment should be ~{self._avg_segment_words()} words

Generate the transcript now. Include segment headers [PRE-GAME], [Q1], etc.
"""
        return prompt

    def _format_speaker_intro(self, announcers: List[str]) -> str:
        """Format speaker introduction section."""
        if not announcers:
            return "BROADCASTERS: Standard broadcast team"

        lines = ["BROADCASTERS:"]
        for i, announcer in enumerate(announcers[:3]):
            role = list(self.speaker_roles.keys())[i] if i < len(self.speaker_roles) else "analyst"
            role_desc = self.speaker_roles.get(role, "")
            lines.append(f"- {announcer} ({role}): {role_desc}")

        return "\n".join(lines)

    def _default_key_moments(self) -> str:
        """Default key moments to generate."""
        moments = []
        for seg in self.segments[:6]:
            moments.append(f"- {seg.name.replace('_', ' ').title()}: {seg.description or seg.content_type}")
        return "\n".join(moments)

    def _default_style_notes(self) -> str:
        """Default style guidance."""
        return f"This is a professional {self.domain} broadcast. Maintain appropriate energy and professionalism."

    def _avg_segment_words(self) -> int:
        """Average words per segment."""
        total_mins = sum(s.duration_minutes for s in self.segments)
        if total_mins == 0:
            return 200
        # Roughly 150 words per minute of broadcast
        return (150 * total_mins) // len(self.segments)


# =============================================================================
# PRE-BUILT DOMAIN TEMPLATES
# =============================================================================

SPORTS_NFL_BROADCAST = DomainTemplate(
    domain="sports",
    event_type="NFL broadcast",
    total_duration_minutes=192,  # ~3h12m realistic broadcast
    speaker_roles={
        "play_by_play": "Handles game action, stats, and transitions (e.g., Kevin Burkhardt, Jim Nantz)",
        "color_analyst": "Former player/coach providing analysis (e.g., Tom Brady, Tony Romo, Troy Aikman)",
        "sideline_reporter": "Field-level updates, injury reports, coach reactions (e.g., Erin Andrews)",
    },
    segments=[
        TranscriptSegment(
            name="pre_game",
            duration_minutes=30,
            content_type="analysis",
            speakers=["analyst", "host"],
            impromptu_ratio=0.35,
            description="Pre-game show with predictions, storylines, and interviews",
        ),
        TranscriptSegment(
            name="q1",
            duration_minutes=38,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.75,  # Play-by-play is mostly reactive/improvised
            description="First quarter action and early game narrative",
        ),
        TranscriptSegment(
            name="q2",
            duration_minutes=38,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.75,
            description="Second quarter with building tension to halftime",
        ),
        TranscriptSegment(
            name="halftime",
            duration_minutes=12,  # Regular NFL halftime is exactly 12 minutes
            content_type="analysis",
            speakers=["analyst", "studio_host"],
            impromptu_ratio=0.4,
            description="Brief halftime analysis from studio crew",
        ),
        TranscriptSegment(
            name="q3",
            duration_minutes=38,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.75,
            description="Third quarter momentum shifts and adjustments",
        ),
        TranscriptSegment(
            name="q4",
            duration_minutes=40,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.8,  # Highest improvisation in clutch moments
            description="Fourth quarter drama, final minutes tension",
        ),
        TranscriptSegment(
            name="post_game",
            duration_minutes=15,
            content_type="analysis",
            speakers=["analyst", "reporter", "sideline_reporter"],
            impromptu_ratio=0.5,
            description="Post-game interviews and quick analysis",
        ),
    ],
    key_moments_prompt="""
1. Opening kickoff - set the scene, crowd energy
2. First touchdown - big reaction moment
3. A spectacular catch or big play - "what a catch!" type moment
4. Halftime transition - summary of first half
5. Fourth quarter with close score - tension building
6. Game-winning play or final seconds - climactic moment
7. Post-game celebration/interviews
""",
    style_notes="""
- Play-by-play is MOSTLY IMPROVISED reactions to live action
- "He throws deep... CAUGHT! At the 30, 25, TOUCHDOWN!"
- Color analyst provides spontaneous analysis: "That's the second time they've run that play-action..."
- Natural banter between announcers fills gaps
- Crowd noise references and atmosphere descriptions
- Stats and historical context woven in naturally during stoppages
- Sideline reporter chimes in for injury updates, coach reactions
""",
)

SPORTS_SUPER_BOWL = DomainTemplate(
    domain="sports",
    event_type="Super Bowl broadcast",
    total_duration_minutes=260,  # ~4h20m for core broadcast (not including 5h pre-game show)
    speaker_roles={
        "play_by_play": "Lead voice calling all game action (e.g., Mike Tirico, Jim Nantz)",
        "color_analyst": "Former player/coach providing expert analysis (e.g., Cris Collinsworth)",
        "sideline_reporter": "Field-level updates, player/coach reactions (e.g., Melissa Stark)",
        "halftime_host": "Entertainment segment host for halftime show",
    },
    segments=[
        # NOTE: Full Super Bowl pre-game is 5+ hours. This covers field ceremonies only.
        TranscriptSegment(
            name="pre_game",
            duration_minutes=45,
            content_type="analysis",
            speakers=["analyst", "host", "sideline_reporter"],
            impromptu_ratio=0.4,
            description="Field ceremonies: national anthem, flyover, coin toss, celebrity sightings",
        ),
        TranscriptSegment(
            name="q1",
            duration_minutes=38,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.75,  # Play-by-play is mostly reactive/improvised
            description="First quarter - establishing game narrative, early momentum",
        ),
        TranscriptSegment(
            name="q2",
            duration_minutes=38,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.75,
            description="Second quarter - building tension toward halftime",
        ),
        TranscriptSegment(
            name="halftime",
            duration_minutes=30,  # Super Bowl halftime is 25-30 min total, performance ~13-15 min
            content_type="entertainment",
            speakers=["halftime_host", "analyst"],
            impromptu_ratio=0.5,
            description="Super Bowl halftime show - performer intro, reactions to spectacle",
        ),
        TranscriptSegment(
            name="q3",
            duration_minutes=38,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.75,
            description="Third quarter - adjustments, momentum shifts",
        ),
        TranscriptSegment(
            name="q4",
            duration_minutes=40,
            content_type="play_by_play",
            speakers=["play_by_play", "color_analyst"],
            impromptu_ratio=0.8,  # Highest improvisation in clutch moments
            description="Fourth quarter drama - everything on the line",
        ),
        TranscriptSegment(
            name="post_game",
            duration_minutes=35,
            content_type="celebration",
            speakers=["analyst", "reporter", "sideline_reporter"],
            impromptu_ratio=0.6,  # High emotion = more improvised
            description="Lombardi trophy presentation, MVP interview, confetti, player families",
        ),
    ],
    key_moments_prompt="""
1. National anthem finish and flyover reaction
2. Opening kickoff - "Here we go, Super Bowl LX!"
3. First big play or touchdown - extended celebration
4. Celebrity crowd shots - noting famous attendees naturally
5. Halftime show intro and key performance moments
6. Fourth quarter tension - crucial plays, clock management
7. Game-winning moment - emotional call
8. Trophy presentation and MVP interview
""",
    style_notes="""
- Super Bowl energy: bigger reactions, more dramatic calls than regular season
- Play-by-play is MOSTLY IMPROVISED reactions to live action
- References to the magnitude of the moment ("This is what they play for!")
- Historical context (past Super Bowls, records, legacy implications)
- Celebrity sightings mentioned naturally during stoppages (Taylor Swift, etc.)
- Halftime: excitement and reactions to spectacle, not knowing specific setlist
- Post-game: high emotion, tears, celebration, legacy discussion
- Crowd shots during timeouts trigger celebrity/fan commentary
""",
)

CORPORATE_EARNINGS_CALL = DomainTemplate(
    domain="corporate",
    event_type="earnings call",
    total_duration_minutes=60,  # 68% of companies run 46-60 min
    speaker_roles={
        "ir_host": "Investor Relations Officer - opens call, reads safe harbor, moderates Q&A",
        "ceo": "Chief Executive Officer - strategic vision, high-level results, handles tough questions",
        "cfo": "Chief Financial Officer - detailed financials, guidance, metrics deep-dive",
    },
    segments=[
        TranscriptSegment(
            name="safe_harbor",
            duration_minutes=3,
            content_type="scripted",
            speakers=["ir_host"],
            impromptu_ratio=0.0,  # 100% scripted legal text
            description="Legal disclaimer, forward-looking statements warning, call logistics",
        ),
        TranscriptSegment(
            name="ceo_remarks",
            duration_minutes=12,
            content_type="scripted",
            speakers=["ceo"],
            impromptu_ratio=0.1,  # Read from prepared script
            description="CEO strategic overview, quarterly highlights, vision",
        ),
        TranscriptSegment(
            name="cfo_financials",
            duration_minutes=12,
            content_type="scripted",
            speakers=["cfo"],
            impromptu_ratio=0.1,
            description="CFO detailed financial review, guidance, key metrics",
        ),
        TranscriptSegment(
            name="analyst_qa",
            duration_minutes=33,  # ~17 questions typical, 1.5-2 min each
            content_type="qa",
            speakers=["ceo", "cfo", "ir_host", "analysts"],
            impromptu_ratio=0.6,  # Most impromptu - market penalizes over-scripted Q&As
            description="Q&A with Wall Street analysts - probing questions, candid answers",
        ),
    ],
    key_moments_prompt="""
1. IR opens with safe harbor statement
2. CEO opening remarks - quarterly highlights, strategic vision
3. Key metric announcements (revenue, EPS, margins)
4. CFO detailed financials and forward guidance
5. Analyst tough questions (margin pressure, competition, macro)
6. CEO/CFO candid responses and deflections
7. Final analyst question and closing remarks
""",
    style_notes="""
- Professional, measured corporate tone
- Specific numbers with precision (millions, percentages to one decimal)
- Industry jargon appropriate to company sector
- Prepared remarks read from script, Q&A more extemporaneous
- AVOID over-scripting Q&A - markets punish this
- Analyst questions can be probing/challenging
- CEO/CFO may deflect: "I'll have to get back to you on that"
- Some dry humor in Q&A acceptable, but restrained
- Executives almost never mention competitors by name (antitrust risk)
- Tesla calls are notably different: Elon ad-libs extensively
""",
)

POLITICS_SPEECH = DomainTemplate(
    domain="politics",
    event_type="political speech",
    total_duration_minutes=45,
    speaker_roles={
        "speaker": "Primary speaker (President, candidate, etc.)",
        "introducer": "Person introducing the main speaker",
    },
    segments=[
        TranscriptSegment(
            name="introduction",
            duration_minutes=5,
            content_type="scripted",
            speakers=["introducer"],
            impromptu_ratio=0.2,
            description="Introduction and crowd warm-up",
        ),
        TranscriptSegment(
            name="opening",
            duration_minutes=5,
            content_type="scripted",
            speakers=["speaker"],
            impromptu_ratio=0.2,
            description="Opening remarks, acknowledgments",
        ),
        TranscriptSegment(
            name="main_body",
            duration_minutes=25,
            content_type="speech",
            speakers=["speaker"],
            impromptu_ratio=0.3,
            description="Main policy points and messaging",
        ),
        TranscriptSegment(
            name="closing",
            duration_minutes=10,
            content_type="speech",
            speakers=["speaker"],
            impromptu_ratio=0.4,
            description="Closing with emotional appeal, call to action",
        ),
    ],
    key_moments_prompt="""
1. Introduction and crowd reaction
2. Opening acknowledgments
3. Key policy announcement
4. Attack on opposition (if campaign)
5. Emotional story or appeal
6. Closing rallying cry
""",
    style_notes="""
- Rhetorical style varies by speaker (formal vs. populist)
- Crowd reactions matter ("cheers", "applause")
- May include ad-libs and crowd interaction
- References to current events
- Repetition of key phrases for emphasis
""",
)

POLITICS_PRESS_BRIEFING = DomainTemplate(
    domain="politics",
    event_type="press briefing",
    total_duration_minutes=40,
    speaker_roles={
        "press_secretary": "White House Press Secretary or equivalent",
        "reporters": "Pool of White House correspondents",
    },
    segments=[
        TranscriptSegment(
            name="opening_statement",
            duration_minutes=5,
            content_type="scripted",
            speakers=["press_secretary"],
            impromptu_ratio=0.1,
            description="Opening remarks and announcements",
        ),
        TranscriptSegment(
            name="qa_session",
            duration_minutes=35,
            content_type="qa",
            speakers=["press_secretary", "reporters"],
            impromptu_ratio=0.7,
            description="Q&A with reporters - highly impromptu",
        ),
    ],
    key_moments_prompt="""
1. Opening statement with key announcements
2. First reporter question on hot topic
3. Follow-up questions pressing for details
4. Contentious exchange with reporter
5. Off-topic question handled
6. Closing "thank you" and exit
""",
    style_notes="""
- More combative tone than other formats
- "I'll have to get back to you on that"
- Deflection and pivot techniques
- Reporters compete for attention
- Real-time news breaking may intrude
""",
)

ENTERTAINMENT_AWARDS = DomainTemplate(
    domain="entertainment",
    event_type="awards show",
    total_duration_minutes=180,
    speaker_roles={
        "host": "Show host - monologue, transitions, category intros",
        "presenter": "Award presenters - category introductions, envelope readers",
        "winner": "Award winners - acceptance speeches (highly variable)",
    },
    segments=[
        TranscriptSegment(
            name="opening_monologue",
            duration_minutes=15,
            content_type="comedy",
            speakers=["host"],
            impromptu_ratio=0.3,
            description="Host's opening monologue with jokes about nominees, current events",
        ),
        TranscriptSegment(
            name="early_awards",
            duration_minutes=45,
            content_type="presentation",
            speakers=["presenter", "winner"],
            impromptu_ratio=0.5,
            description="Technical and supporting categories",
        ),
        TranscriptSegment(
            name="musical_performance",
            duration_minutes=10,
            content_type="entertainment",
            speakers=["host"],
            impromptu_ratio=0.2,
            description="Musical number or entertainment break",
        ),
        TranscriptSegment(
            name="major_awards",
            duration_minutes=60,
            content_type="presentation",
            speakers=["presenter", "winner"],
            impromptu_ratio=0.6,
            description="Major categories - actor, director, picture",
        ),
        TranscriptSegment(
            name="in_memoriam",
            duration_minutes=5,
            content_type="scripted",
            speakers=["host"],
            impromptu_ratio=0.1,
            description="In Memoriam tribute",
        ),
        TranscriptSegment(
            name="finale",
            duration_minutes=15,
            content_type="presentation",
            speakers=["presenter", "winner", "host"],
            impromptu_ratio=0.5,
            description="Best Picture and closing",
        ),
    ],
    key_moments_prompt="""
1. Host's opening jokes about nominees and current events
2. Surprise winner reaction
3. Emotional acceptance speech with thank-yous
4. Play-off music incident (speech runs long)
5. Political statement or controversial moment
6. Best Picture announcement and reaction
""",
    style_notes="""
- Mix of scripted banter and genuine emotion
- Winners frequently go off-script in acceptance speeches
- Host's jokes reference current events and nominees
- Some speeches rambling, some polished and prepared
- Occasional technical difficulties or envelope mishaps
- Political statements in speeches are common
""",
)


# =============================================================================
# OLYMPIC TEMPLATES
# =============================================================================

SPORTS_OLYMPIC_OPENING = DomainTemplate(
    domain="sports",
    event_type="Olympic Opening Ceremony",
    total_duration_minutes=240,  # ~4 hours typical
    speaker_roles={
        "main_host": "Primetime host providing narrative flow (e.g., Mike Tirico)",
        "celebrity_cohost": "Entertainment perspective, emotional reactions (e.g., Kelly Clarkson, Peyton Manning)",
        "route_reporter": "On-site reporters covering parade from different vantage points",
        "cultural_expert": "Explains artistic performance meanings and cultural context",
    },
    segments=[
        TranscriptSegment(
            name="pre_show",
            duration_minutes=30,
            content_type="analysis",
            speakers=["main_host", "celebrity_cohost"],
            impromptu_ratio=0.5,  # Scene-setting allows ad-lib
            description="Build-up, scene-setting, cultural context, host banter",
        ),
        TranscriptSegment(
            name="parade_of_nations",
            duration_minutes=120,  # 90-120 min for 200+ nations
            content_type="commentary",
            speakers=["main_host", "celebrity_cohost", "route_reporter"],
            impromptu_ratio=0.4,  # Scripted country intros but improvised reactions
            description="All delegations enter - prepared stories plus live reactions",
        ),
        TranscriptSegment(
            name="artistic_performances",
            duration_minutes=60,
            content_type="commentary",
            speakers=["main_host", "cultural_expert"],
            impromptu_ratio=0.3,  # Explain meanings but react to spectacle
            description="Host nation cultural showcase, artistic acts",
        ),
        TranscriptSegment(
            name="official_protocol",
            duration_minutes=20,
            content_type="scripted",
            speakers=["main_host"],
            impromptu_ratio=0.1,  # Formal, scripted
            description="IOC speeches, organizing committee, head of state",
        ),
        TranscriptSegment(
            name="cauldron_lighting",
            duration_minutes=30,
            content_type="ceremony",
            speakers=["main_host", "celebrity_cohost"],
            impromptu_ratio=0.4,  # Emotional reactions allowed
            description="Final torch relay, dramatic cauldron lighting",
        ),
    ],
    key_moments_prompt="""
1. Opening spectacle and first artistic performance
2. Host nation delegation enters (biggest cheers)
3. Notable athletes spotted (flag bearers, stars)
4. Celebrity sightings in crowd
5. Most memorable artistic moment
6. IOC President speech highlights
7. Final torch bearer reveal and cauldron lighting
""",
    style_notes="""
- Mixture of prepared country facts and spontaneous reactions
- Celebrity co-hosts add entertainment perspective and emotional reactions
- Ad-lib freedom during parade transitions between countries
- Artistic performances: explain cultural meaning while reacting to spectacle
- Celebrity sightings mentioned naturally during transitions
- Cauldron lighting is emotional peak - allow genuine reactions
- NBC uses multiple reporters at different locations
""",
)

SPORTS_OLYMPIC_CLOSING = DomainTemplate(
    domain="sports",
    event_type="Olympic Closing Ceremony",
    total_duration_minutes=180,  # ~3 hours, shorter than opening
    speaker_roles={
        "main_host": "Primetime host (e.g., Mike Tirico)",
        "celebrity_cohost": "Entertainment perspective",
        "reporter": "On-site reactions and interviews",
    },
    segments=[
        TranscriptSegment(
            name="athletes_parade",
            duration_minutes=30,
            content_type="celebration",
            speakers=["main_host", "celebrity_cohost"],
            impromptu_ratio=0.7,  # Very festive, athletes mingling
            description="Athletes enter mixed together (not by nation), celebratory mood",
        ),
        TranscriptSegment(
            name="artistic_program",
            duration_minutes=50,
            content_type="entertainment",
            speakers=["main_host", "celebrity_cohost"],
            impromptu_ratio=0.4,
            description="Celebratory performances, less formal than opening",
        ),
        TranscriptSegment(
            name="marathon_medals",
            duration_minutes=15,
            content_type="ceremony",
            speakers=["main_host"],
            impromptu_ratio=0.3,
            description="Marathon medal ceremony (traditional final event)",
        ),
        TranscriptSegment(
            name="flag_handover",
            duration_minutes=15,
            content_type="ceremony",
            speakers=["main_host"],
            impromptu_ratio=0.2,
            description="Antwerp ceremony - flag passed to next host city",
        ),
        TranscriptSegment(
            name="next_host_presentation",
            duration_minutes=15,
            content_type="entertainment",
            speakers=["main_host", "celebrity_cohost"],
            impromptu_ratio=0.5,  # React to spectacle (e.g., Tom Cruise stunt)
            description="Next host city showcase (LA28, etc.)",
        ),
        TranscriptSegment(
            name="closing_speeches",
            duration_minutes=10,
            content_type="scripted",
            speakers=["main_host"],
            impromptu_ratio=0.1,
            description="IOC President declares Games closed",
        ),
        TranscriptSegment(
            name="flame_extinguishing",
            duration_minutes=10,
            content_type="ceremony",
            speakers=["main_host", "celebrity_cohost"],
            impromptu_ratio=0.5,  # Emotional finale
            description="Cauldron extinguished - emotional conclusion",
        ),
        TranscriptSegment(
            name="post_ceremony",
            duration_minutes=35,
            content_type="celebration",
            speakers=["main_host", "reporter"],
            impromptu_ratio=0.7,  # Party mode
            description="Concert, celebration, athlete interactions",
        ),
    ],
    key_moments_prompt="""
1. Athletes entering together, dancing, taking selfies
2. Notable athlete moments (dancing, celebrating)
3. Marathon medal ceremony
4. Flag handover to next host city
5. Next host city presentation spectacle
6. IOC President's closing declaration
7. Flame extinguishing - emotional moment
8. Post-ceremony celebration highlights
""",
    style_notes="""
- More relaxed and festive than opening ceremony
- Athletes mingle freely - lots of spontaneous moments
- Higher ad-lib freedom due to celebratory atmosphere
- Next host presentation often has surprise element (stunts, celebrities)
- Flame extinguishing is bittersweet emotional peak
- Post-ceremony is party mode - very impromptu
""",
)


# =============================================================================
# POLITICAL TEMPLATES (ADDITIONAL)
# =============================================================================

POLITICS_STATE_OF_THE_UNION = DomainTemplate(
    domain="politics",
    event_type="State of the Union address",
    total_duration_minutes=150,  # Pre-coverage + speech + response + analysis
    speaker_roles={
        "anchor": "Network anchor hosting coverage (e.g., Lester Holt, David Muir)",
        "president": "President of the United States delivering address",
        "opposition_responder": "Opposition party member delivering response",
        "analyst": "Political analysts providing commentary",
    },
    segments=[
        TranscriptSegment(
            name="pre_coverage",
            duration_minutes=30,
            content_type="analysis",
            speakers=["anchor", "analyst"],
            impromptu_ratio=0.5,
            description="Preview of themes, congressional arrivals, analysis",
        ),
        TranscriptSegment(
            name="entrance",
            duration_minutes=5,
            content_type="ceremony",
            speakers=["anchor"],
            impromptu_ratio=0.3,  # Describe scene, handshakes
            description="President enters chamber, works the aisle",
        ),
        TranscriptSegment(
            name="speech",
            duration_minutes=75,  # 60-90 min typical (includes applause)
            content_type="address",
            speakers=["president"],
            impromptu_ratio=0.05,  # Heavily telepromptered, rare ad-libs
            description="President's address (70-130 applause interruptions)",
        ),
        TranscriptSegment(
            name="opposition_response",
            duration_minutes=10,
            content_type="scripted",
            speakers=["opposition_responder"],
            impromptu_ratio=0.05,  # Pre-recorded or heavily scripted
            description="Opposition party official response",
        ),
        TranscriptSegment(
            name="post_analysis",
            duration_minutes=30,
            content_type="analysis",
            speakers=["anchor", "analyst"],
            impromptu_ratio=0.6,
            description="Fact-checking, spin interpretation, key moments",
        ),
    ],
    key_moments_prompt="""
1. President's entrance and aisle interactions
2. Opening remarks and first applause
3. Key policy announcement
4. Partisan disruption (if any - e.g., "You lie!" moments)
5. Emotional guest recognition in gallery
6. Closing rhetoric
7. Opposition response key points
8. Analyst debate on implications
""",
    style_notes="""
- During speech: networks stay mostly quiet, show crowd reactions
- Applause breaks allow brief commentary or cutaways
- Real-time fact-checking increasingly common
- Partisan disruptions are rare but memorable
- Opposition response often delivered from different location
- Post-analysis can get contentious between analysts
""",
)

POLITICS_DEBATE = DomainTemplate(
    domain="politics",
    event_type="presidential debate",
    total_duration_minutes=150,  # 90 min debate + pre/post coverage
    speaker_roles={
        "moderator": "Debate moderator - asks questions, enforces rules",
        "candidate_a": "First candidate (by party/incumbent)",
        "candidate_b": "Second candidate (challenger)",
        "analyst": "Post-debate analysts in spin room",
    },
    segments=[
        TranscriptSegment(
            name="pre_debate",
            duration_minutes=20,
            content_type="analysis",
            speakers=["anchor", "analyst"],
            impromptu_ratio=0.5,
            description="Stakes preview, what each candidate needs to accomplish",
        ),
        TranscriptSegment(
            name="debate",
            duration_minutes=90,  # Standard format
            content_type="debate",
            speakers=["moderator", "candidate_a", "candidate_b"],
            impromptu_ratio=0.4,  # Prepared answers but follow-ups unpredictable
            description="Structured debate with 2-min responses and rebuttals",
        ),
        TranscriptSegment(
            name="spin_room",
            duration_minutes=40,
            content_type="analysis",
            speakers=["analyst", "surrogates"],
            impromptu_ratio=0.7,  # Highly reactive to what just happened
            description="Post-debate spin room, surrogate interviews, instant polls",
        ),
    ],
    key_moments_prompt="""
1. Opening statements (if format includes)
2. First major policy clash
3. Personal attack and response
4. Moderator follow-up that surprises candidate
5. Memorable zinger or gaffe
6. Closing statements
7. Post-debate spin room reactions
""",
    style_notes="""
- 2024 format: muted mics when not candidate's turn
- No live audience in recent debates
- Moderators may fact-check in real-time (controversial)
- Candidates prepare heavily but follow-ups cause improvisation
- Spin room is highly impromptu - campaigns push narratives
- Key moments often come in latter half when candidates tire
- Post-debate: first 2-3 hours most critical for narrative
""",
)

POLITICS_RALLY = DomainTemplate(
    domain="politics",
    event_type="campaign rally",
    total_duration_minutes=90,  # Main speaker portion only
    speaker_roles={
        "speaker": "Main candidate/political figure",
        "warm_up": "Warm-up speakers (local politicians, celebrities)",
    },
    segments=[
        TranscriptSegment(
            name="warm_up",
            duration_minutes=10,  # Just representing main speaker entry
            content_type="introduction",
            speakers=["warm_up"],
            impromptu_ratio=0.4,
            description="Final warm-up speaker, candidate introduction",
        ),
        TranscriptSegment(
            name="opening",
            duration_minutes=10,
            content_type="speech",
            speakers=["speaker"],
            impromptu_ratio=0.5,
            description="Candidate takes stage, acknowledges crowd, local references",
        ),
        TranscriptSegment(
            name="greatest_hits",
            duration_minutes=30,
            content_type="speech",
            speakers=["speaker"],
            impromptu_ratio=0.65,  # Signature riffs delivered spontaneously
            description="Core campaign themes, signature lines, crowd favorites",
        ),
        TranscriptSegment(
            name="policy_tangents",
            duration_minutes=25,
            content_type="speech",
            speakers=["speaker"],
            impromptu_ratio=0.7,  # Often goes off teleprompter here
            description="Policy discussion, attacks on opponents, current events",
        ),
        TranscriptSegment(
            name="closing",
            duration_minutes=15,
            content_type="speech",
            speakers=["speaker"],
            impromptu_ratio=0.6,  # Rambling more common as energy fades
            description="Closing rally cry, call to action, exit to music",
        ),
    ],
    key_moments_prompt="""
1. Candidate takes stage to crowd roar
2. Local reference or shout-out
3. Signature campaign line (delivered spontaneously)
4. Attack on opponent
5. Policy tangent that goes off-script
6. Crowd interaction moment
7. Closing rallying cry
""",
    style_notes="""
- 60-70% of rally speeches are improvised/ad-libbed
- Teleprompter present but frequently ignored
- Highest off-script risk: 45-60 minutes in
- "Greatest hits" (border wall, "fake news", etc.) delivered spontaneously
- Attacks on opponents are most ad-lib heavy
- Breaking news often incorporated same-day
- Energy drops in final third - more rambling/circular
- Full events run 3-5 hours (doors open to end) but main speaker ~90 min
""",
)


# =============================================================================
# TEMPLATE REGISTRY AND DETECTION
# =============================================================================

TEMPLATE_REGISTRY: Dict[str, DomainTemplate] = {
    # Sports - NFL
    "nfl_broadcast": SPORTS_NFL_BROADCAST,
    "super_bowl": SPORTS_SUPER_BOWL,
    "sports_nfl": SPORTS_NFL_BROADCAST,
    "sports_super_bowl": SPORTS_SUPER_BOWL,
    # Sports - Olympics
    "olympic_opening": SPORTS_OLYMPIC_OPENING,
    "olympic_closing": SPORTS_OLYMPIC_CLOSING,
    "olympics_opening": SPORTS_OLYMPIC_OPENING,
    "olympics_closing": SPORTS_OLYMPIC_CLOSING,
    "sports_olympic_opening": SPORTS_OLYMPIC_OPENING,
    "sports_olympic_closing": SPORTS_OLYMPIC_CLOSING,
    # Corporate
    "earnings_call": CORPORATE_EARNINGS_CALL,
    "corporate_earnings": CORPORATE_EARNINGS_CALL,
    # Politics - General
    "speech": POLITICS_SPEECH,
    "political_speech": POLITICS_SPEECH,
    "press_briefing": POLITICS_PRESS_BRIEFING,
    "politics_speech": POLITICS_SPEECH,
    # Politics - Specific Events
    "state_of_the_union": POLITICS_STATE_OF_THE_UNION,
    "sotu": POLITICS_STATE_OF_THE_UNION,
    "debate": POLITICS_DEBATE,
    "presidential_debate": POLITICS_DEBATE,
    "rally": POLITICS_RALLY,
    "campaign_rally": POLITICS_RALLY,
    # Entertainment
    "awards": ENTERTAINMENT_AWARDS,
    "awards_show": ENTERTAINMENT_AWARDS,
    "entertainment_awards": ENTERTAINMENT_AWARDS,
}


def detect_template(event_ticker: str, event_title: str) -> DomainTemplate:
    """Detect the appropriate template based on event ticker and title.

    Args:
        event_ticker: Kalshi event ticker (e.g., "KXSBLX")
        event_title: Event title for additional context

    Returns:
        Best-matching DomainTemplate
    """
    ticker_upper = event_ticker.upper()
    title_upper = event_title.upper() if event_title else ""

    # Super Bowl detection (highest priority for NFL)
    if "SB" in ticker_upper or "SUPER BOWL" in title_upper or "SUPERBOWL" in title_upper:
        return SPORTS_SUPER_BOWL

    # Olympics detection (check ticker patterns too)
    is_olympics = (
        "OLYMPIC" in title_upper or
        "OLYMPICS" in title_upper or
        "OLY" in ticker_upper or
        ("PARIS" in title_upper and "CEREMONY" in title_upper) or
        ("MILAN" in title_upper and "CEREMONY" in title_upper) or
        ("LA" in title_upper and "2028" in title_upper) or
        ("BEIJING" in title_upper and "CEREMONY" in title_upper)
    )
    if is_olympics:
        if "CLOSING" in title_upper:
            return SPORTS_OLYMPIC_CLOSING
        else:
            # Default to opening ceremony for Olympics
            return SPORTS_OLYMPIC_OPENING

    # State of the Union detection
    if "STATE OF THE UNION" in title_upper or "SOTU" in ticker_upper:
        return POLITICS_STATE_OF_THE_UNION

    # Debate detection
    if "DEBATE" in title_upper:
        return POLITICS_DEBATE

    # Rally detection
    if "RALLY" in title_upper:
        return POLITICS_RALLY

    # NFL detection
    if "NFL" in ticker_upper or "FOOTBALL" in title_upper:
        return SPORTS_NFL_BROADCAST

    # Earnings call detection
    if "EARNINGS" in title_upper or "CALL" in title_upper or any(
        co in title_upper for co in ["TESLA", "APPLE", "GOOGLE", "META", "EA", "NVDA", "AMAZON", "MICROSOFT"]
    ):
        return CORPORATE_EARNINGS_CALL

    # Political speech detection (generic - after specific types)
    if any(pol in title_upper for pol in ["TRUMP", "BIDEN", "PRESIDENT", "SPEECH", "ADDRESS"]):
        return POLITICS_SPEECH

    # Press briefing detection
    if "PRESS" in title_upper or "BRIEFING" in title_upper or "WHITE HOUSE" in title_upper:
        return POLITICS_PRESS_BRIEFING

    # Awards show detection
    if any(award in title_upper for award in ["OSCAR", "GRAMMY", "EMMY", "GOLDEN GLOBE", "AWARDS", "ACADEMY"]):
        return ENTERTAINMENT_AWARDS

    # Default to NFL broadcast (most common mentions market domain)
    logger.warning(f"Could not detect template for {event_ticker}, defaulting to NFL broadcast")
    return SPORTS_NFL_BROADCAST


def get_template(template_name: str) -> Optional[DomainTemplate]:
    """Get a template by name from the registry.

    Args:
        template_name: Template key (e.g., "super_bowl", "earnings_call")

    Returns:
        DomainTemplate or None if not found
    """
    return TEMPLATE_REGISTRY.get(template_name.lower())


def list_templates() -> List[str]:
    """List all available template names."""
    return list(TEMPLATE_REGISTRY.keys())
