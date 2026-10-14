"""Offline checks for the evidence-first, two-source workflow."""

import asyncio
import io
import json
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from verifier import (CVFacts, FieldCheck, SourceReport, VerificationPipeline,
                      _hard_linkedin_checks, _hard_facebook_checks, _structured,
                      score_cvs_impl, score_evidence)


def report():
    return SourceReport(**{field: FieldCheck(status="match", evidence="Agrees")
                           for field in ("name", "city", "jobs", "education", "skills")})


FACTS = CVFacts.model_validate({
    "name": "Alex Chan", "city": "London", "industry": "Design",
    "jobs": [{"company": "Acme Ltd", "title": "Senior Designer",
              "start_year": 2020, "is_current": True}],
    "education": [{"school": "Local University", "degree": "BSc",
                   "field": "Design", "graduation_year": 2019}],
    "skills": ["UI/UX Design", "Prototyping"],
})
LINKEDIN = {
    "id": 101, "name": "Alex Chan", "city": "London", "country": "UK",
    "industry": "Design", "experience": [
        {"company": "Acme", "title": "Designer", "seniority": "senior",
         "start_year": 2020, "end_year": None, "is_current": True}],
    "education": [{"school": "Local University", "degree": "Bachelor of Science",
                   "field": "Design", "end_year": 2019}],
    "skills": [{"name": "UI/UX"}, {"name": "Prototyping"}],
}
FACEBOOK = {
    "id": 908, "display_name": "Lex Chan", "original_name": "Alex Chan",
    "city": "London", "current_company": "Acme", "current_job": "Designer",
    "education": "Local University, BSc Design",
}


class FakeTool:
    def __init__(self, name, output, order):
        self.name, self.output, self.order = name, output, order

    async def ainvoke(self, arguments):
        self.order.append((self.name, arguments))
        value = self.output(arguments) if callable(self.output) else self.output
        return [{"type": "text", "text": json.dumps(value)}]


class FakeAgent:
    def __init__(self, value, label, order):
        self.value, self.label, self.order = value, label, order

    async def ainvoke(self, inputs, config):
        self.order.append((self.label, inputs["messages"][0]["content"]))
        return {"structured_response": self.value, "messages": inputs["messages"]}


def fake_pipeline(order, facebook=True):
    pipeline = object.__new__(VerificationPipeline)
    pipeline.tool_limit = asyncio.Semaphore(3)
    pipeline.tools = {
        "search_linkedin_people": FakeTool("search_linkedin_people", [
            {"id": 102, "name": "Alex Chan", "location": "London, UK", "industry": "Design"},
            {"id": 101, "name": "Alex Chan", "location": "London, UK", "industry": "Design"}], order),
        "get_linkedin_profile": FakeTool("get_linkedin_profile", lambda a:
            LINKEDIN if a["person_id"] == 101 else {
                **LINKEDIN, "id": 102, "experience": [], "education": [], "skills": []}, order),
        "search_facebook_users": FakeTool("search_facebook_users", [
            {"id": 908, "display_name": "Lex Chan", "city": "London"}
        ] if facebook else [], order),
        "get_facebook_profile": FakeTool("get_facebook_profile", FACEBOOK, order),
    }
    pipeline.extractor = FakeAgent(FACTS, "extract", order)
    pipeline.linkedin_auditor = FakeAgent(report(), "linkedin_audit", order)
    pipeline.facebook_auditor = FakeAgent(report(), "facebook_audit", order)
    return pipeline


class RuleTests(unittest.TestCase):
    def test_exact_and_equivalent_facts(self):
        li, fb = report(), report()
        _hard_linkedin_checks(FACTS, LINKEDIN, li)
        _hard_facebook_checks(FACTS, FACEBOOK, fb)
        self.assertTrue(all(getattr(li, field).status == "match"
                            for field in ("name", "city", "jobs", "education", "skills")))
        self.assertEqual(fb.jobs.status, "match")
        self.assertEqual(fb.skills.status, "unavailable")
        self.assertEqual(fb.education.status, "match")

    def test_single_false_year_seniority_or_extra_skill(self):
        for edit in ("year", "seniority", "skill"):
            facts = FACTS.model_copy(deep=True)
            if edit == "year":
                facts.jobs[0].start_year = 2021
            elif edit == "seniority":
                facts.jobs[0].title = "Junior Designer"
            else:
                facts.skills.append("Docker")
            li = report()
            _hard_linkedin_checks(facts, LINKEDIN, li)
            self.assertEqual(getattr(li, "skills" if edit == "skill" else "jobs").status,
                             "mismatch")

    def test_two_sources_and_conflicting_evidence(self):
        evidence = {"identity": {"confident": True}, "linkedin_profile": LINKEDIN,
                    "facebook_profile": FACEBOOK, "linkedin_report": report().model_dump(),
                    "facebook_report": report().model_dump()}
        self.assertEqual(score_evidence(evidence), 0.95)
        evidence["linkedin_report"]["jobs"]["status"] = "mismatch"
        evidence["facebook_report"]["jobs"]["status"] = "unavailable"
        self.assertEqual(score_evidence(evidence), 0.05)
        evidence["facebook_report"]["jobs"]["status"] = "match"
        self.assertEqual(score_evidence(evidence), 0.05)
        evidence["linkedin_report"]["jobs"]["status"] = "match"
        evidence["facebook_report"]["jobs"]["status"] = "mismatch"
        self.assertEqual(score_evidence(evidence), 0.5)
        evidence["facebook_profile"] = None
        self.assertEqual(score_evidence(evidence), 0.5)
        evidence["facebook_profile"] = {**FACEBOOK, "current_company": "Other"}
        self.assertEqual(score_evidence(evidence), 0.5)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_cv_city_still_reaches_unfiltered_identity_search(self):
        order = []
        pipeline = fake_pipeline(order)
        facts = FACTS.model_copy(deep=True)
        facts.city = "Wrong City"
        def search(arguments):
            if "location" not in arguments:
                return [{"id": 101, "name": "Alex Chan", "location": "London, UK",
                         "industry": "Design"}]
            start = 200 if "industry" in arguments else 300
            return [{"id": start + i, "name": "Alex Chan", "location": "Wrong City",
                     "industry": "Design"} for i in range(8)]
        pipeline.tools["search_linkedin_people"] = FakeTool(
            "search_linkedin_people", search, order)
        pipeline.tools["get_linkedin_profile"] = FakeTool(
            "get_linkedin_profile", lambda a: LINKEDIN if a["person_id"] == 101 else {
                **LINKEDIN, "id": a["person_id"], "city": "Wrong City",
                "experience": [], "education": [], "skills": []}, order)
        profile, confident, _ = await pipeline._linkedin(facts, [])
        self.assertTrue(confident)
        self.assertEqual(profile["id"], 101)
        self.assertTrue(any(name == "search_linkedin_people" and "location" not in args
                            for name, args in order))

    async def test_capped_nickname_search_requires_profile_anchors(self):
        order = []
        pipeline = fake_pipeline(order, facebook=False)
        pipeline.tools["get_facebook_profile"] = FakeTool(
            "get_facebook_profile", lambda a: {**FACEBOOK, "id": a["user_id"]}, order)
        profile, confident, _ = await pipeline._facebook(FACTS, LINKEDIN, [])
        self.assertTrue(confident)
        self.assertEqual(profile["id"], LINKEDIN["id"])
        self.assertEqual(order[0][0], "search_facebook_users")
        pipeline.tools["get_facebook_profile"] = FakeTool(
            "get_facebook_profile", lambda a: {
                **FACEBOOK, "id": a["user_id"], "original_name": "Another Person",
                "current_company": "Other Corp"}, [])
        _, confident, _ = await pipeline._facebook(FACTS, LINKEDIN, [])
        self.assertFalse(confident)

    async def test_both_profiles_fetched_before_ordered_comparisons(self):
        order = []
        result = await fake_pipeline(order).verify("a sample CV")
        self.assertEqual(result["linkedin_profile"]["id"], 101)
        self.assertEqual(result["facebook_profile"]["id"], 908)
        self.assertEqual(score_evidence(result), 0.95)
        labels = [event[0] for event in order]
        self.assertLess(labels.index("get_facebook_profile"), labels.index("linkedin_audit"))
        self.assertLess(labels.index("linkedin_audit"), labels.index("facebook_audit"))

    async def test_missing_facebook_is_uncertain(self):
        result = await fake_pipeline([], facebook=False).verify("a sample CV")
        self.assertIsNone(result["facebook_profile"])
        self.assertEqual(score_evidence(result), 0.5)

    async def test_plain_text_output_gets_schema_reminder(self):
        class PlainThenStructured:
            calls = 0
            async def ainvoke(self, inputs, config):
                self.calls += 1
                if self.calls == 1:
                    return {"messages": inputs["messages"], "structured_response": None}
                self.reminder = inputs["messages"][-1]["content"]
                return {"messages": inputs["messages"], "structured_response": FACTS}
        agent = PlainThenStructured()
        value = await _structured(agent, CVFacts, {"cv_text": "data"})
        self.assertEqual(value.name, "Alex Chan")
        self.assertEqual(agent.calls, 2)
        self.assertIn("CVFacts", agent.reminder)

    async def test_one_cv_failure_does_not_abort_or_exceed_three(self):
        class Agent:
            active = peak = 0
            async def ainvoke(self, inputs):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.01)
                    if "FAIL" in inputs["cv_text"]:
                        raise RuntimeError("private request body")
                    return {"identity": {"confident": True}, "linkedin_profile": LINKEDIN,
                            "facebook_profile": FACEBOOK,
                            "linkedin_report": report().model_dump(),
                            "facebook_report": report().model_dump()}
                finally:
                    self.active -= 1
        agent = Agent()
        cvs = {f"test-{i}.pdf": "data" for i in range(8)}
        cvs["broken.pdf"] = "FAIL"
        captured = io.StringIO()
        with patch.object(__import__("verifier").Path, "write_text"), redirect_stderr(captured):
            scores = await score_cvs_impl(agent, cvs)
        self.assertEqual(scores["broken.pdf"], 0.5)
        self.assertTrue(all(score == 0.95 for name, score in scores.items()
                            if name != "broken.pdf"))
        self.assertLessEqual(agent.peak, 3)
        self.assertNotIn("private request body", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
