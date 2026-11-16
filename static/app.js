const form = document.querySelector("#question-form");
const question = document.querySelector("#question");
const submit = document.querySelector("#submit");
const requestStatus = document.querySelector("#request-status");
const result = document.querySelector("#result");
const title = document.querySelector("#result-title");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const sourceList = document.querySelector("#source-list");
const language = document.querySelector("#language");
const theme = document.querySelector("#theme");
const dock = document.querySelector("#examples-dock");
const motionDocks = document.querySelectorAll("dock-wrapper");

const translations = {
  fr: {
    title: "FinRAG · Assistant BIAT",
    intro: "Les rapports de gestion de la BIAT, avec les passages à l’appui.",
    years: "2021, 2023 et 2025",
    language: "Langue", theme: "Thème", light: "Clair", dark: "Sombre",
    question: "Votre question", passages: "Passages par rapport", submit: "Interroger",
    examples: "Questions à essayer",
    incomeLabel: "Résultat 2024", comparisonLabel: "Comparer le PNB", employeesLabel: "Effectif 2023",
    income: "Quel était le résultat net de la BIAT en 2024 ?",
    comparison: "Compare le produit net bancaire de la banque seule en 2021, 2023 et 2025.",
    employees: "Quel est l’effectif total de la BIAT en 2023 ?",
    sources: "Passages consultés",
    sourceNote: "Les extraits restent en français. Les citations indiquent l’année du rapport et la page PDF.",
    table: "Tableau de comparaison",
    required: "Saisissez une question.",
    searching: "Recherche…",
    waiting: "Recherche dans les rapports… Le premier chargement peut prendre un moment.",
    unexpected: "Le serveur a renvoyé une réponse inattendue.",
    network: "Connexion au serveur impossible. Réessayez.",
    status: {answer: "Réponse", refusal: "Information indisponible", failure: "Réponse indisponible"},
  },
  en: {
    title: "FinRAG · BIAT assistant",
    intro: "Explore BIAT’s management reports with citations to the original passages.",
    years: "2021, 2023 and 2025",
    language: "Language", theme: "Theme", light: "Light", dark: "Dark",
    question: "Your question", passages: "Passages per report", submit: "Ask",
    examples: "Questions to try",
    incomeLabel: "2024 net income", comparisonLabel: "Compare NBI", employeesLabel: "2023 headcount",
    income: "What was BIAT’s net income in 2024?",
    comparison: "Compare BIAT’s standalone net banking income in 2021, 2023 and 2025.",
    employees: "What was BIAT’s headcount in 2023?",
    sources: "Retrieved passages",
    sourceNote: "Excerpts stay in French. Citations show the report year and PDF page.",
    table: "Comparison table",
    required: "Enter a question.",
    searching: "Searching…",
    waiting: "Searching the reports… The first request can take a moment.",
    unexpected: "The server returned an unexpected response.",
    network: "Could not connect to the server. Try again.",
    status: {answer: "Answer", refusal: "Information unavailable", failure: "Answer unavailable"},
  },
};

let copy = translations.fr;
let selectedLanguage = "fr";

function readPreference(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function savePreference(key, value) {
  try { localStorage.setItem(key, value); } catch { /* The current selection still works. */ }
}

function updateLanguage() {
  copy = translations[selectedLanguage];
  document.documentElement.lang = selectedLanguage;
  document.title = copy.title;
  for (const element of document.querySelectorAll("[data-i18n]")) {
    element.textContent = copy[element.dataset.i18n];
  }
  question.placeholder = copy.income;
  question.setCustomValidity("");
  if (result.dataset.state) title.textContent = copy.status[result.dataset.state];
  for (const region of answer.querySelectorAll(".table-wrap")) region.setAttribute("aria-label", copy.table);
}

const savedLanguage = readPreference("finrag-language");
selectedLanguage = savedLanguage === "en" ? "en" : "fr";
language.querySelector(`input[value="${selectedLanguage}"]`).checked = true;
language.addEventListener("change", event => {
  selectedLanguage = event.target.value;
  savePreference("finrag-language", selectedLanguage);
  updateLanguage();
});
updateLanguage();

const savedTheme = readPreference("finrag-theme");
const initialTheme = savedTheme === "dark" || savedTheme === "light"
  ? savedTheme : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
theme.querySelector(`input[value="${initialTheme}"]`).checked = true;
document.documentElement.dataset.theme = initialTheme;
theme.addEventListener("change", event => {
  document.documentElement.dataset.theme = event.target.value;
  savePreference("finrag-theme", event.target.value);
});

const dockMotion = matchMedia("(hover: hover) and (pointer: fine) and (min-width: 32rem) and (prefers-reduced-motion: no-preference)");
function updateDockMotion() {
  for (const wrapper of motionDocks) wrapper.toggleAttribute("disabled", !dockMotion.matches);
}
dockMotion.addEventListener("change", updateDockMotion);
updateDockMotion();

function appendText(container, text, citations) {
  for (const part of text.split(/(\[\d{4} p\. \d+\])/g)) {
    const source = citations.get(part);
    if (!source) {
      container.append(part);
      continue;
    }
    const link = document.createElement("a");
    link.href = "#" + source.id;
    link.textContent = part;
    link.addEventListener("click", event => {
      event.preventDefault();
      source.open = true;
      source.querySelector("summary").focus({preventScroll: true});
      source.scrollIntoView({block: "nearest"});
    });
    container.append(link);
  }
}

function tableCells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map(cell => cell.trim());
}

function appendAnswer(text, citations) {
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length;) {
    const headers = tableCells(lines[i]);
    const separators = tableCells(lines[i + 1] ?? "");
    const isTable = headers.length > 1 && separators.length === headers.length
      && separators.every(cell => /^:?-{3,}:?$/.test(cell));
    if (!isTable) {
      appendText(answer, lines[i], citations);
      if (++i < lines.length) answer.append("\n");
      continue;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "table-wrap";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", copy.table);
    const table = document.createElement("table");
    const header = table.createTHead().insertRow();
    for (const value of headers) {
      const cell = document.createElement("th");
      cell.scope = "col";
      appendText(cell, value, citations);
      header.append(cell);
    }
    const body = table.createTBody();
    i += 2;
    while (i < lines.length) {
      const values = tableCells(lines[i]);
      if (values.length !== headers.length) break;
      const row = body.insertRow();
      for (const value of values) appendText(row.insertCell(), value, citations);
      i++;
    }
    wrapper.append(table);
    answer.append(wrapper);
  }
}

function showResult(status, text, passages = []) {
  result.dataset.state = status;
  title.textContent = copy.status[status];
  answer.lang = selectedLanguage;
  answer.replaceChildren();
  sourceList.replaceChildren();
  const citations = new Map();

  passages.forEach((passage, index) => {
    const source = document.createElement("details");
    source.id = "source-" + (index + 1);
    const summary = document.createElement("summary");
    summary.textContent = passage.citation;
    const excerpt = document.createElement("p");
    excerpt.lang = "fr";
    excerpt.textContent = passage.text;
    source.append(summary, excerpt);
    citations.set(passage.citation, source);
    sourceList.append(source);
  });

  appendAnswer(text, citations);
  sources.hidden = !passages.length;
  result.hidden = false;
  title.focus();
}

for (const example of dock.querySelectorAll("[data-example]")) {
  example.addEventListener("click", () => {
    question.value = copy[example.dataset.example];
    question.setCustomValidity("");
    question.focus();
  });
}

question.addEventListener("input", () => question.setCustomValidity(""));

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (submit.disabled) return;
  const value = question.value.trim();
  if (!value) {
    question.setCustomValidity(copy.required);
    question.reportValidity();
    return;
  }

  submit.disabled = true;
  language.disabled = true;
  submit.textContent = copy.searching;
  result.hidden = true;
  requestStatus.textContent = copy.waiting;
  try {
    const response = await fetch(form.dataset.endpoint, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: value, top_k: Number(form.elements["top-k"].value), language: selectedLanguage}),
    });
    const isJson = response.headers.get("content-type")?.includes("application/json");
    const data = isJson ? await response.json() : null;
    if (!response.ok) throw new Error(data?.error || copy.unexpected);
    if (!data || !Object.hasOwn(copy.status, data.status) || typeof data.answer !== "string" || !Array.isArray(data.sources)) {
      throw new Error(copy.unexpected);
    }
    showResult(data.status, data.answer, data.sources);
  } catch (error) {
    const message = error instanceof TypeError ? copy.network : error.message;
    showResult("failure", message);
  } finally {
    requestStatus.textContent = "";
    submit.disabled = false;
    language.disabled = false;
    submit.textContent = copy.submit;
  }
});
