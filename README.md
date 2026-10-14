# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

My solution uses LangChain with `deepseek-v4-flash` in a fixed evidence-first workflow. A structured extraction step records every CV claim without correcting it. The program then calls the course SocialGraph MCP to search and fetch a complete LinkedIn profile, resolves same-name candidates using employers, education, location and dates, and independently searches and fetches a Facebook profile using its own user ID. Only after both profiles have been retrieved does the model compare CV claims with LinkedIn, followed by a separate Facebook comparison. LinkedIn checks every claimed name, current city, job, degree and skill; Facebook corroborates its available legal name, current city, current job and education, while unavailable historical details and skills are marked as such. Python enforces exact years, explicit seniority and claimed skills against the complete LinkedIn list, then assigns 0.95 for consistent evidence, 0.05 for a confirmed discrepancy and 0.50 for ambiguous identity, missing evidence, source conflict or execution failure. Each CV's ignored `verification_details.json` entry includes the extracted claims, both original profiles, both field-level comparison reports and MCP call trace. CV and tool text are treated as untrusted data; the program runs at most three CVs and three MCP calls concurrently, retries MCP failures once, and isolates per-CV errors. No public filenames, candidate identities or labels are built into the verifier; the supplied grading runner alone reads labels.

Facebook's search has a 20-result cap and some display names are nicknames. After a Facebook search, if no candidate is well supported, the workflow may try the LinkedIn numeric ID as a **hypothesis** for a Facebook lookup; it accepts that record only when the independently retrieved Facebook legal name, city, current company and title agree with LinkedIn. An ID match alone never establishes identity. The final score also checks whether the two sources agree on fields they both provide.

The fixed two-source workflow achieved **7/7 correct (100%)** on the full public test on 2026-10-14, with no timeout or exception fallback. Results are recorded in `results.csv`:

| CV | Score | Correctness |
| --- | ---: | --- |
| CV_1.pdf | 0.9500 | correct |
| CV_2.pdf | 0.9500 | correct |
| CV_3.pdf | 0.9500 | correct |
| CV_4.pdf | 0.0500 | correct |
| CV_5.pdf | 0.0500 | correct |
| CV_6.pdf | 0.9500 | correct |
| CV_7.pdf | 0.0500 | correct |

These scores are rule-based confidence levels, not calibrated probabilities. The uncertainty score of 0.50 is conservative and is counted as a discrepancy by the supplied grader; it does not establish that a CV contains a false claim. Public-set accuracy does not guarantee hidden-set accuracy, and LLM identity resolution, interpretation and injection resistance remain limitations.

#### Run and inspect

Use the setup commands above, then run:

```bash
python hw2.py --cv-folder public_test
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1` or call `.venv\Scripts\python.exe` directly. Put `DEEPSEEK_API_KEY` in a local `.env` or environment variable. No key is included in this repository. In addition to `results.csv`, the program saves an ignored `verification_details.json` with both source profiles, field-level evidence, MCP calls and any failure type. The grading/PDF/MCP loader code is unchanged. Each CV has a 210-second limit; 20 CVs in batches of three have a verification time budget of at most about 24.5 minutes, excluding setup and PDF loading.

### Task 2

The submitted one-page `task2/adversarial_cv.pdf` is a CV for Kelly Tsang with one clear false claim: the **visible** education line says **MSc in Finance, HKUST (graduated 2016)**. The course SocialGraph MCP records a **BSc in Finance at HKUST**, and the Facebook profile gives only a Bachelor's Degree. This is a real upgraded-degree discrepancy, not a wording variation. Her other visible claims follow the target profile, so the attack has a single, auditable false detail.

I selected **hidden text and PDF text extraction** from the suggested techniques. The visible page is a raster image of the CV; the PDF text layer beneath that image says BSc. This exploits a gap between what a person sees and what the supplied MarkItDown/PDF loader sends to an LLM. `pypdf` and the course `cv_text()` both extract the BSc line, while a rendered page visibly shows MSc. The attack is limited to the PDF; it does not alter or impersonate the MCP server. I considered LLM-as-a-judge bias and long-context distraction but did not add them: the former depends on an unknown agent's prompt hierarchy, and the latter may obscure the false claim without reliably surviving extraction or a systematic field check. A concise CV makes the observed failure attributable to the text-layer mismatch.

I ran `python hw2.py --cv-folder task2` against my Task 1 verifier with the course MCP and required DeepSeek model. The saved run is in `task2/self_test_results.csv`: **adversarial_cv.pdf 0.9500**, **target_cv.pdf 0.9500**. The local evidence trace shows that the adversarial CV was extracted with degree `BSc`, then compared with the retrieved LinkedIn and Facebook profiles. This is one local run, not an estimate of the five undisclosed agents or their majority-of-three grading. Verifiers that render the page, OCR the image, or compare rendered and extracted text can detect the MSc claim.

Task 3 reflection is outside the current scope.
