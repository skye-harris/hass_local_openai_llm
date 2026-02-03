import express from "express";
import fetch from "node-fetch";

const app = express();
const PORT = 9090;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const defaultBaseUrl = process.argv[2] || '';

async function weaviateFetch(base, path, apiKey, options = {}) {
  const res = await fetch(`${base}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      ...(options.headers || {}),
    },
  });
  if (!res.ok) {
  	throw new Error(await res.text());
  }
  return res.json();
}

/* ---------- Routes ---------- */

app.get("/", (_, res) => res.send(renderPage()));

app.post("/list", async (req, res) => {
  try {
    const data = await weaviateFetch(req.body.base, "/v1/graphql", req.body.apiKey, {
      method: "POST",
      body: JSON.stringify({
        query: `{
          Get {
            ${req.body.class}(
              limit: 20
              sort: [{ path: ["_creationTimeUnix"], order: desc }]
            ) {
              query
              content
              _additional { id }
            }
          }
        }`,
      }),
    });
    res.json(data.data.Get[req.body.class] ?? []);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/add", async (req, res) => {
  try {
    await weaviateFetch(req.body.base, "/v1/objects", req.body.apiKey, {
      method: "POST",
      body: JSON.stringify({
        class: req.body.class,
        properties: {
          query: req.body.query,
          content: req.body.content,
        },
      }),
    });
    res.sendStatus(200);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/delete", async (req, res) => {
  try {
    await weaviateFetch(req.body.base, `/v1/objects/${req.body.id}`, req.body.apiKey, {
      method: "DELETE",
    });
    res.sendStatus(200);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/search", async (req, res) => {
	console.log(req.body)
  const gql =
    req.body.type === "hybrid"
      ? `{
          Get {
            ${req.body.class}(
              hybrid: {
                query: "${req.body.query}"
                properties: ["query"]
                alpha: ${req.body.alpha}
				bm25SearchOperator: {
				  operator: Or
				  minimumOrTokensMatch: 2
				}
              }
              limit: 5
            ) {
              query
              content
              _additional {
	              score
				  rerank(
					property: "query"
					query: "Reorder the results so that the most relevant data is first, using the following request to guide you: ${req.body.query}"
				  ) {
					score
				  }
              }
            }
          }
        }`
      : `{
          Get {
            ${req.body.class}(
              nearText: { concepts: ["${req.body.query}"], certainty: 0.5 }
              limit: 5
            ) {
              content
              _additional {
              	certainty
				  rerank(
					property: "content"
				  ) {
					score
				  }
              }
            }
          }
        }`;
  try {
    const data = await weaviateFetch(req.body.base, "/v1/graphql", req.body.apiKey, {
      method: "POST",
      body: JSON.stringify({ query: gql }),
    });
    console.log(JSON.stringify(data))
    const results = data.data.Get[req.body.class] ?? []
    console.log(results)
    res.json(results);
  } catch (e) {
    console.log(e)

    res.status(500).json({ error: e.message });
  }
});

app.post("/classes", async (req, res) => {
  const { base } = req.body;
  if (!base) return res.status(400).json({ error: "Missing base URL" });

  try {
    const result = await fetch(`${base}/v1/graphql`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${req.body.apiKey}`,
      },
      body: JSON.stringify({
        query: `
          {
            __schema {
              types {
                name
                fields {
                  name
                }
              }
            }
          }
        `,
      }),
    });

    if (!result.ok) throw new Error(await result.text());
    const json = await result.json();

	const types = json.data.__schema.types.find((type) => type.name === "GetObjectsObj")?.fields
    res.json(types?.map((type) => type.name));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


/* ---------- UI ---------- */

function renderPage() {
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Weaviate RAG Admin</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
:root {
  --bg1: #1b2040;
  --bg2: #0f1220;
  --panel: #171a2b;
  --accent: #6cf2c2;
  --muted: #9aa0c3;
}

* { box-sizing: border-box }

html {
  min-height: 100%;
}

body {
  min-height: 100%;
  margin: 0;
  font-family: system-ui;
  color: #fff;
  background: linear-gradient(180deg, var(--bg1), var(--bg2));
  background-attachment: fixed;
}

header {
  max-width: 1200px;
  margin: 0 auto;
  padding: 30px 20px 10px;
}

.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px 30px;
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 20px;
}

.panel {
  background: linear-gradient(180deg, #1a1e35, var(--panel));
  border-radius: 14px;
  padding: 16px;
}

.sidebar {
  align-self: start;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.sidebar > div {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

h1 { margin: 0 }
h2 { margin: 0 }

input, textarea, button, select {
  background: #101324;
  border: 1px solid #2a2f55;
  color: #fff;
  padding: 10px;
  border-radius: 8px;
}

button {
  background: linear-gradient(135deg, var(--accent), #58d8ff);
  color: #000;
  font-weight: 600;
  cursor: pointer;
  align-self: flex-start;
}

button:disabled { opacity: .5 }

.col { display: flex; flex-direction: column; gap: 10px }
.row { display: flex; gap: 10px; align-items: center }

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.list {
  max-height: 360px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.card {
  background: #0f1328;
  padding: 12px;
  border-radius: 10px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 10px;
}

.small { font-size: 12px; color: var(--muted) }

.results > div {
  background: #0f1328;
  padding: 8px;
  border-radius: 8px;
}

.right-align {
  text-align: right;
}

.bold {
	font-weight: bold;
}

#scoreThreshold, #hybridAlpha {
  width: 100px;
}
</style>
</head>

<body>

<header>
  <h1>Weaviate RAG Admin</h1>
</header>

<div class="app">

  <!-- LEFT -->
  <div class="col">

    <form id="searchForm">
    <div class="panel col">
      <h2>Search</h2>
      <div class="row">
      	<label class="col" style="flex:1">
      		Query
        	<input id="searchInput" placeholder="Search…">
        </label>
		<label class="col">
      		Threshold
        	<input id="scoreThreshold" type="number" value="0.95">
        </label>
		<label class="col">
      		Alpha
        	<input id="hybridAlpha" type="number" value="0.5">
        </label>
      </div>
      <div class="row right-align" style="margin-top: 4px">
        <button onclick="runSearch()">Go</button>
      </div>
      <div id="searchResults" class="results col"></div>
    </div>
    </form>

    <div class="panel col">
      <div class="section-header">
        <h2>Stored Content</h2>
        <button onclick="refreshList()">Refresh</button>
      </div>

      <div id="list" class="list"></div>

      <hr style="opacity:.2">

      <h3>Add Entry</h3>
      <input id="addQuery" placeholder="Query">
      <textarea id="addContent" rows="3" placeholder="Content"></textarea>
      <button id="addBtn" onclick="addItem()">Add</button>
    </div>

  </div>

  <!-- SIDEBAR -->
  <div class="panel sidebar">
  	<div>
		<h2>Server</h2>
		<input id="baseUrl" placeholder="http://localhost:8080">
		<input id="apiKey" placeholder="API Key">
    </div>

	<div>
		<h2>Object Class</h2>
		<select id="objectClass">
		</select>
    </div>
  </div>

</div>

<script>
const baseInput = document.getElementById('baseUrl');
const objectClasses = document.getElementById("objectClass");
const apiKeyInput = document.getElementById("apiKey");
//const searchType = document.getElementById("searchType");
const hybridAlpha = document.getElementById('hybridAlpha');
const scoreThreshold = document.getElementById('scoreThreshold');
const searchForm = document.getElementById('searchForm');

baseInput.value = "${defaultBaseUrl}" || localStorage.getItem('weaviateBase') || '';
apiKeyInput.value = localStorage.getItem('weaviateApiKey') || '';

baseInput.addEventListener('change', async () => {
  localStorage.setItem('weaviateBase', baseInput.value);
  await refreshClasses();
});

objectClasses.addEventListener('change', async () => {
  await refreshList();
});

apiKeyInput.addEventListener('change', async () => {
  localStorage.setItem('weaviateApiKey', apiKeyInput.value);
  await refreshClasses();
});

const base = () => baseInput.value.trim();

async function refreshList() {
  const res = await fetch('/list', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ base: base(), class: objectClasses.value, apiKey: apiKeyInput.value })
  });

  const items = await res.json();
  const list = document.getElementById('list');
  list.innerHTML = '';

  items.forEach(i => {
    const card = document.createElement('div');
    card.className = 'card';

    const left = document.createElement('div');

    const id = document.createElement('div');
    id.className = 'small';
    id.textContent = i._additional.id;

    const q = document.createElement('strong');
    q.textContent = i.query;

    const c = document.createElement('div');
    c.textContent = i.content;

    left.append(id, q, c);

    const del = document.createElement('button');
    del.textContent = '✕';
    del.onclick = () => delItem(i._additional.id);

    card.append(left, del);
    list.appendChild(card);
  });
}

async function refreshClasses() {
  const res = await fetch('/classes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ base: base(), apiKey: apiKeyInput.value })
  });

  if (!res.ok) {
  	console.log(JSON.stringify(res))
  	return;
  }

  const items = await res.json();

  objectClasses.innerHTML = "";

  // populate select
  items.forEach(item => {
    const option = document.createElement("option");
    option.value = item;
    option.textContent = item;
    objectClasses.appendChild(option);
  });

  await refreshList();
}

async function addItem() {
  addBtn.disabled = true;
  await fetch('/add', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      base: base(),
      query: addQuery.value,
      content: addContent.value,
      class: objectClasses.value,
      apiKey: apiKeyInput.value
    })
  });
  addQuery.value = '';
  addContent.value = '';
  addBtn.disabled = false;
  refreshList();
}

async function delItem(id) {
  await fetch('/delete', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ base: base(), id, apiKey: apiKeyInput.value })
  });
  refreshList();
}

async function runSearch(event) {
  const res = await fetch('/search', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      base: base(),
//      type: searchType.value,
	  type: "hybrid",
	  alpha: hybridAlpha.value,
      query: searchInput.value,
      class: objectClasses.value,
      apiKey: apiKeyInput.value
    })
  });

  if (!res.ok) {
  	console.error(res);
  	return;
  }

  const data = await res.json();
  const out = document.getElementById('searchResults');
  out.innerHTML = '';

  data = data.filter((datum) => {
  	return Number(datum._additional.score ?? datum._additional.certainty) >= Number(scoreThreshold.value);
  }).forEach(i => {
	const result = document.createElement('div');
	result.className = 'col';
	result.style.gap = 'none';

	const queryRow = document.createElement('div');
	const contentRow = document.createElement('div');
	const scoreRow = document.createElement('div');

	const score = i._additional.score ?? i._additional.certainty;

	queryRow.textContent = \`\${i.query}\`;
	contentRow.textContent = \`\${i.content}\`;
	scoreRow.textContent = \`Score: \${score}\`;

	queryRow.className = 'bold';

	for (let ele of [queryRow,  contentRow, scoreRow])
		result.appendChild(ele)

	out.appendChild(result);
  });
}

searchForm.addEventListener('submit', (event) => {
	event.preventDefault();
	runSearch();
	return false;
});

refreshClasses();
</script>

</body>
</html>`;
}

app.listen(PORT, () => {
  console.log(`UI running at http://localhost:${PORT}`);
});
