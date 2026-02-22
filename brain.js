const createDebug = require('debug');

const debug = createDebug('app:brain');

const STOPWORDS = new Set([
  'the',
  'and',
  'that',
  'this',
  'with',
  'from',
  'your',
  'you',
  'are',
  'was',
  'were',
  'have',
  'has',
  'had',
  'for',
  'but',
  'not',
  'all',
  'can',
  'how',
  'what',
  'when',
  'where',
  'why',
  'who',
  'about',
  'into',
  'then',
  'than',
  'just',
  'like',
  'they',
  'them',
  'their',
  'our',
  'out',
  'will',
  'would',
  'could',
  'should',
  'been',
  'being'
]);

const MATH_PRECEDENCE = {
  '+': 1,
  '-': 1,
  '*': 2,
  '/': 2,
  '^': 3
};

const RIGHT_ASSOCIATIVE = new Set(['^']);

class ChatbotBrain {
  constructor(store, options = {}) {
    this.store = store;
    this.maxMemory = options.maxMemory || 5000;
    const envSearchWindow = Number(process.env.MEMORY_SEARCH_WINDOW || 1500);
    this.memorySearchWindow = Number.isFinite(envSearchWindow) && envSearchWindow > 0 ? envSearchWindow : 1500;
    const envTemp = Number(process.env.ESSAY_TEMPERATURE);
    this.essayTemperature = Number.isFinite(envTemp) ? Math.min(1, Math.max(0, envTemp)) : 0.65;
    this.neuralDim = 48;
  }

  ensureNeuralState() {
    if (!this.store.state.neural || typeof this.store.state.neural !== 'object') {
      this.store.state.neural = {
        dim: this.neuralDim,
        trainedSamples: 0,
        lastTrainAt: null,
        prototypes: []
      };
    }

    if (!Array.isArray(this.store.state.neural.prototypes)) {
      this.store.state.neural.prototypes = [];
    }

    if (!Number.isFinite(Number(this.store.state.neural.dim))) {
      this.store.state.neural.dim = this.neuralDim;
    }

    return this.store.state.neural;
  }

  hashTokenToIndex(token, dim) {
    const text = String(token || '');
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = (hash * 16777619) >>> 0;
    }
    return hash % dim;
  }

  buildNeuralVector(text, dim) {
    const tokens = this.tokenize(text);
    const vector = new Array(dim).fill(0);
    if (!tokens.length) {
      return vector;
    }

    for (const token of tokens) {
      const idx = this.hashTokenToIndex(token, dim);
      vector[idx] += 1;
    }

    const magnitude = Math.sqrt(vector.reduce((sum, item) => sum + item * item, 0));
    if (magnitude <= 0) {
      return vector;
    }

    return vector.map((value) => value / magnitude);
  }

  cosineSimilarity(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
      return 0;
    }

    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let index = 0; index < a.length; index += 1) {
      const av = Number(a[index]) || 0;
      const bv = Number(b[index]) || 0;
      dot += av * bv;
      normA += av * av;
      normB += bv * bv;
    }

    if (normA <= 0 || normB <= 0) {
      return 0;
    }

    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  shouldSkipNeuralTraining(interaction, options = {}) {
    const force = Boolean(options.force);
    if (!interaction || !interaction.user || !interaction.bot) {
      return true;
    }

    if (force) {
      const user = String(interaction.user || '').trim();
      const bot = String(interaction.bot || '').trim();
      return !user || !bot;
    }

    const source = String(interaction.source || '');
    if (
      [
        'web_ingest',
        'starter_bootstrap',
        'concept_association',
        'generated',
        'clarification',
        'knowledge_gap'
      ].includes(source)
    ) {
      return true;
    }

    if (this.isLowSignalInput(interaction.user)) {
      return true;
    }

    const bot = String(interaction.bot || '');
    if (!bot.trim() || this.isPollutedWebText(bot) || bot.length > 1400) {
      return true;
    }

    return false;
  }

  trainNeuralFromInteraction(interaction, options = {}) {
    if (this.shouldSkipNeuralTraining(interaction, options)) {
      return;
    }

    const neural = this.ensureNeuralState();
    const dim = Number(neural.dim) || this.neuralDim;
    const inputVector = this.buildNeuralVector(interaction.user, dim);
    const replyText = String(interaction.bot || '').trim();

    let target = null;
    let bestReplySimilarity = 0;
    for (const prototype of neural.prototypes) {
      const replySimilarity = this.scoreSimilarity(replyText, prototype.reply || '');
      if (replySimilarity > bestReplySimilarity) {
        bestReplySimilarity = replySimilarity;
        target = prototype;
      }
    }

    if (!target || bestReplySimilarity < 0.78) {
      neural.prototypes.push({
        reply: replyText,
        source: String(interaction.source || 'neural_memory'),
        vector: inputVector,
        count: 1,
        updatedAt: new Date().toISOString()
      });
    } else {
      const oldCount = Number(target.count) || 1;
      const learningRate = Math.max(0.08, 1 / (oldCount + 1));
      target.vector = target.vector.map((value, index) => {
        const incoming = Number(inputVector[index]) || 0;
        return value * (1 - learningRate) + incoming * learningRate;
      });
      target.count = oldCount + 1;
      target.updatedAt = new Date().toISOString();

      if ((target.reply || '').length < 50 && replyText.length > target.reply.length) {
        target.reply = replyText;
      }
    }

    neural.trainedSamples = (Number(neural.trainedSamples) || 0) + 1;
    neural.lastTrainAt = new Date().toISOString();

    if (neural.prototypes.length > 900) {
      neural.prototypes = neural.prototypes
        .sort((left, right) => (Number(right.count) || 0) - (Number(left.count) || 0))
        .slice(0, 900);
    }
  }

  importAllMemoryToNeural() {
    const neural = this.ensureNeuralState();
    neural.prototypes = [];
    neural.trainedSamples = 0;
    neural.lastTrainAt = null;

    let importedSamples = 0;
    const seenPairs = new Set();
    const maxInteractionImportsRaw = Number(process.env.NEURAL_IMPORT_MAX_INTERACTIONS || 120000);
    const maxInteractionImports = Number.isFinite(maxInteractionImportsRaw) && maxInteractionImportsRaw > 0
      ? Math.floor(maxInteractionImportsRaw)
      : 120000;

    const importPair = (user, bot, source = 'memory_import', confidence = 0.8) => {
      const userText = String(user || '').trim();
      const botText = String(bot || '').trim();
      if (!userText || !botText) {
        return;
      }

      const pairKey = `${source}\u0000${userText.slice(0, 500)}\u0000${botText.slice(0, 500)}`;
      if (seenPairs.has(pairKey)) {
        return;
      }
      seenPairs.add(pairKey);

      if (this.isLowSignalInput(userText) || this.isPollutedWebText(botText) || botText.length > 1600) {
        return;
      }

      this.trainNeuralFromInteraction(
        {
          user: userText,
          bot: botText,
          source,
          confidence
        },
        { force: false }
      );
      importedSamples += 1;
    };

    const interactions = Array.isArray(this.store.state.interactions) ? this.store.state.interactions : [];
    const interactionSliceStart = Math.max(0, interactions.length - maxInteractionImports);
    for (let index = interactionSliceStart; index < interactions.length; index += 1) {
      const item = interactions[index];
      importPair(item?.user, item?.bot, item?.source || 'interaction_memory', item?.confidence || 0.7);
    }

    const responseBank = this.store.state.responseBank || {};
    for (const [normalizedPrompt, entries] of Object.entries(responseBank)) {
      if (!Array.isArray(entries)) {
        continue;
      }
      for (const entry of entries) {
        importPair(normalizedPrompt, entry?.reply, 'response_bank', 0.72);
      }
    }

    const learnedFacts = this.store.state.learnedFacts || {};
    for (const fact of Object.keys(learnedFacts)) {
      importPair(`remember that ${fact}`, fact, 'learned_fact', 0.74);
      importPair(`what is ${fact}`, fact, 'learned_fact', 0.74);
    }

    const webKnowledge = this.store.state.webKnowledge || {};
    for (const [url, item] of Object.entries(webKnowledge)) {
      const title = String(item?.title || 'Untitled Page').trim();
      const summary = String(item?.lastSummary || '').trim();
      importPair(`[WEB:${url}] ${title}`, summary, 'web_knowledge', 0.78);
    }

    this.store.scheduleSave(['neural']);

    return {
      importedSamples,
      trainedSamples: Number(neural.trainedSamples) || 0,
      neuralPrototypes: Array.isArray(neural.prototypes) ? neural.prototypes.length : 0,
      neuralDim: Number(neural.dim) || this.neuralDim
    };
  }

  async importAllMemoryToNeuralAsync(options = {}) {
    const neural = this.ensureNeuralState();
    neural.prototypes = [];
    neural.trainedSamples = 0;
    neural.lastTrainAt = null;

    const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
    const yieldEveryRaw = Number(options.yieldEvery || process.env.NEURAL_IMPORT_YIELD_EVERY || 400);
    const yieldEvery = Number.isFinite(yieldEveryRaw) && yieldEveryRaw > 0 ? Math.floor(yieldEveryRaw) : 400;
    const maxInteractionImportsRaw = Number(process.env.NEURAL_IMPORT_MAX_INTERACTIONS || 120000);
    const maxInteractionImports = Number.isFinite(maxInteractionImportsRaw) && maxInteractionImportsRaw > 0
      ? Math.floor(maxInteractionImportsRaw)
      : 120000;

    let importedSamples = 0;
    let consideredSamples = 0;
    const seenPairs = new Set();

    const maybeYield = async () => {
      if (consideredSamples > 0 && consideredSamples % yieldEvery === 0) {
        if (onProgress) {
          onProgress({
            consideredSamples,
            importedSamples,
            trainedSamples: Number(neural.trainedSamples) || 0,
            neuralPrototypes: Array.isArray(neural.prototypes) ? neural.prototypes.length : 0
          });
        }
        await new Promise((resolve) => setImmediate(resolve));
      }
    };

    const importPair = async (user, bot, source = 'memory_import', confidence = 0.8) => {
      consideredSamples += 1;

      const userText = String(user || '').trim();
      const botText = String(bot || '').trim();
      if (!userText || !botText) {
        await maybeYield();
        return;
      }

      const pairKey = `${source}\u0000${userText.slice(0, 500)}\u0000${botText.slice(0, 500)}`;
      if (seenPairs.has(pairKey)) {
        await maybeYield();
        return;
      }
      seenPairs.add(pairKey);

      if (this.isLowSignalInput(userText) || this.isPollutedWebText(botText) || botText.length > 1600) {
        await maybeYield();
        return;
      }

      this.trainNeuralFromInteraction(
        {
          user: userText,
          bot: botText,
          source,
          confidence
        },
        { force: false }
      );
      importedSamples += 1;
      await maybeYield();
    };

    const interactions = Array.isArray(this.store.state.interactions) ? this.store.state.interactions : [];
    const interactionSliceStart = Math.max(0, interactions.length - maxInteractionImports);
    for (let index = interactionSliceStart; index < interactions.length; index += 1) {
      const item = interactions[index];
      await importPair(item?.user, item?.bot, item?.source || 'interaction_memory', item?.confidence || 0.7);
    }

    const responseBank = this.store.state.responseBank || {};
    for (const [normalizedPrompt, entries] of Object.entries(responseBank)) {
      if (!Array.isArray(entries)) {
        continue;
      }
      for (const entry of entries) {
        await importPair(normalizedPrompt, entry?.reply, 'response_bank', 0.72);
      }
    }

    const learnedFacts = this.store.state.learnedFacts || {};
    for (const fact of Object.keys(learnedFacts)) {
      await importPair(`remember that ${fact}`, fact, 'learned_fact', 0.74);
      await importPair(`what is ${fact}`, fact, 'learned_fact', 0.74);
    }

    const webKnowledge = this.store.state.webKnowledge || {};
    for (const [url, item] of Object.entries(webKnowledge)) {
      const title = String(item?.title || 'Untitled Page').trim();
      const summary = String(item?.lastSummary || '').trim();
      await importPair(`[WEB:${url}] ${title}`, summary, 'web_knowledge', 0.78);
    }

    this.store.scheduleSave(['neural']);

    if (onProgress) {
      onProgress({
        done: true,
        consideredSamples,
        importedSamples,
        trainedSamples: Number(neural.trainedSamples) || 0,
        neuralPrototypes: Array.isArray(neural.prototypes) ? neural.prototypes.length : 0
      });
    }

    return {
      importedSamples,
      trainedSamples: Number(neural.trainedSamples) || 0,
      neuralPrototypes: Array.isArray(neural.prototypes) ? neural.prototypes.length : 0,
      neuralDim: Number(neural.dim) || this.neuralDim
    };
  }

  buildNeuralCandidate(message) {
    const neural = this.ensureNeuralState();
    if (!neural.prototypes.length) {
      return null;
    }

    const dim = Number(neural.dim) || this.neuralDim;
    const inputVector = this.buildNeuralVector(message, dim);

    let best = null;
    for (let index = 0; index < neural.prototypes.length; index += 1) {
      const prototype = neural.prototypes[index];
      if (!prototype || !Array.isArray(prototype.vector) || !prototype.reply) {
        continue;
      }

      const cosine = this.cosineSimilarity(inputVector, prototype.vector);
      if (!best || cosine > best.cosine) {
        best = {
          prototype,
          index,
          cosine
        };
      }
    }

    if (!best || best.cosine < 0.5) {
      return null;
    }

    const normalizedReply = this.normalize(best.prototype.reply);
    if (/jump to content\s+main menu/i.test(normalizedReply) || this.isPollutedWebText(best.prototype.reply)) {
      return null;
    }

    const confidence = this.clamp(0.56 + best.cosine * 0.36, 0.56, 0.97);
    return {
      text: best.prototype.reply,
      source: 'neural_memory',
      baseScore: confidence,
      neuralRef: {
        prototypeIndex: best.index,
        cosine: Number(best.cosine.toFixed(3)),
        count: Number(best.prototype.count) || 1
      }
    };
  }

  normalize(text) {
    return String(text || '')
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  isPollutedWebText(text) {
    const sample = this.normalize(text);
    if (!sample) {
      return true;
    }

    const badSignals = [
      'jump to content',
      'main menu',
      'move to sidebar',
      'create account',
      'log in personal tools',
      'special pages search',
      'toggle history subsection'
    ];

    const hits = badSignals.filter((signal) => sample.includes(signal)).length;
    return hits >= 2;
  }

  tokenize(text) {
    return this.normalize(text)
      .split(' ')
      .filter((token) => token.length > 1);
  }

  extractConcepts(text) {
    const tokens = this.tokenize(text);
    return [...new Set(tokens.filter((token) => token.length > 2 && !STOPWORDS.has(token)))].slice(0, 16);
  }

  ensureSession(sessionId) {
    if (!this.store.state.sessions[sessionId]) {
      this.store.state.sessions[sessionId] = {
        turns: 0,
        firstSeen: new Date().toISOString(),
        lastSeen: new Date().toISOString(),
        recentTurns: [],
        pendingClarification: null
      };
      this.store.state.stats.sessions += 1;
    }

    if (!Array.isArray(this.store.state.sessions[sessionId].recentTurns)) {
      this.store.state.sessions[sessionId].recentTurns = [];
    }

    if (!this.store.state.sessions[sessionId].pendingClarification) {
      this.store.state.sessions[sessionId].pendingClarification = null;
    }

    this.store.state.sessions[sessionId].lastSeen = new Date().toISOString();
  }

  rememberSessionTurn(sessionId, user, bot, source = 'generated') {
    if (!sessionId || !this.store.state.sessions[sessionId]) {
      return;
    }

    const userText = String(user || '').trim();
    const botText = String(bot || '').trim();
    if (!userText || !botText) {
      return;
    }

    const session = this.store.state.sessions[sessionId];
    session.recentTurns.push({
      user: userText,
      bot: botText,
      source: String(source || 'generated'),
      at: new Date().toISOString()
    });

    if (session.recentTurns.length > 6) {
      session.recentTurns = session.recentTurns.slice(-6);
    }
  }

  getLastContextAnchorTurn(sessionId) {
    const session = this.store.state.sessions[sessionId];
    if (!session || !Array.isArray(session.recentTurns) || session.recentTurns.length === 0) {
      return null;
    }

    for (let index = session.recentTurns.length - 1; index >= 0; index -= 1) {
      const turn = session.recentTurns[index];
      if (!turn || !turn.user) {
        continue;
      }

      const source = String(turn.source || '');
      if (source === 'clarification_needed' || source === 'knowledge_gap') {
        continue;
      }

      if (this.isLowSignalInput(turn.user)) {
        continue;
      }

      return turn;
    }

    return null;
  }

  buildContextAwareQuery(sessionId, message) {
    const session = this.store.state.sessions[sessionId];
    if (!session || !Array.isArray(session.recentTurns) || session.recentTurns.length === 0) {
      return String(message || '').trim();
    }

    const rawMessage = String(message || '').trim();
    const normalized = this.normalize(rawMessage);

    const whatAboutMatch = normalized.match(/^what about (.+)$/);
    if (whatAboutMatch) {
      const subject = String(whatAboutMatch[1] || '')
        .replace(/^(an|a|the)\s+/, '')
        .trim();

      if (subject && subject.length >= 2) {
        return `define ${subject}`;
      }
      return rawMessage;
    }

    const looksFollowup = /(it|that|this|they|them|those|these|he|she|why|how|explain more|what about|and then|continue)/.test(
      normalized
    );
    const explicitFollowup = /(like i said|as i said|from above|from before|based on that|using that|what i said|the previous one|above)/.test(
      normalized
    );

    const tokenCount = this.tokenize(rawMessage).length;
    const shortReference = tokenCount <= 5 && /(it|that|this|they|them|those|these|same|again|more|continue|then)/.test(normalized);
    const directNewTopicPrompt = /^(define|what is|whats|what's|who is|tell me about|explain|describe)\b/.test(
      normalized
    );

    if (directNewTopicPrompt && !session.pendingClarification) {
      return rawMessage;
    }

    if (!looksFollowup && !explicitFollowup && !shortReference && !session.pendingClarification) {
      return rawMessage;
    }

    const anchorTurn =
      (session.pendingClarification && session.pendingClarification.anchorUser
        ? { user: session.pendingClarification.anchorUser }
        : null) || this.getLastContextAnchorTurn(sessionId);

    if (!anchorTurn || !anchorTurn.user) {
      return rawMessage;
    }

    const overlap = this.scoreSimilarity(anchorTurn.user, rawMessage);
    if (overlap >= 0.72) {
      return rawMessage;
    }

    if (tokenCount >= 4 && overlap < 0.18 && !session.pendingClarification) {
      return rawMessage;
    }

    return `${anchorTurn.user} ${rawMessage}`;
  }

  updateTokenGraph(tokens) {
    if (!tokens.length) {
      return;
    }

    const tokenGraph = this.store.state.tokenGraph;
    const sequence = ['<START>', ...tokens, '<END>'];

    for (let index = 0; index < sequence.length - 1; index += 1) {
      const from = sequence[index];
      const to = sequence[index + 1];

      if (!tokenGraph[from]) {
        tokenGraph[from] = {};
      }

      tokenGraph[from][to] = (tokenGraph[from][to] || 0) + 1;
    }
  }

  rememberResponse(message, reply) {
    if (this.isLowSignalInput(message)) {
      return;
    }

    const replyText = String(reply || '');
    if (/\(I used a similar memory from past chats\.\)/i.test(replyText)) {
      return;
    }
    if (/jump to content\s+main menu/i.test(this.normalize(replyText))) {
      return;
    }
    if (/(another related source says|node\.js.? download archive|very low trust score|scam warnings online)/i.test(replyText)) {
      return;
    }
    if (replyText.length > 1400) {
      return;
    }

    const key = this.normalize(message);
    if (!key) {
      return;
    }

    if (!this.store.state.responseBank[key]) {
      this.store.state.responseBank[key] = [];
    }

    const existing = this.store.state.responseBank[key].find((entry) => entry.reply === reply);
    if (existing) {
      existing.count += 1;
      existing.lastUsed = new Date().toISOString();
    } else {
      this.store.state.responseBank[key].push({
        reply,
        count: 1,
        lastUsed: new Date().toISOString()
      });
    }

    this.store.state.responseBank[key] = this.store.state.responseBank[key]
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }

  updateConceptGraph(concepts) {
    if (!concepts.length) {
      return;
    }

    const conceptGraph = this.store.state.conceptGraph;
    for (const concept of concepts) {
      if (!conceptGraph[concept]) {
        conceptGraph[concept] = { count: 0, lastSeen: null };
      }
      conceptGraph[concept].count += 1;
      conceptGraph[concept].lastSeen = new Date().toISOString();
    }
  }

  updateAssociationGraph(concepts) {
    if (concepts.length < 2) {
      return;
    }

    const associationGraph = this.store.state.associationGraph;

    for (let leftIndex = 0; leftIndex < concepts.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < concepts.length; rightIndex += 1) {
        const left = concepts[leftIndex];
        const right = concepts[rightIndex];

        if (!associationGraph[left]) {
          associationGraph[left] = {};
        }
        if (!associationGraph[right]) {
          associationGraph[right] = {};
        }

        associationGraph[left][right] = (associationGraph[left][right] || 0) + 1;
        associationGraph[right][left] = (associationGraph[right][left] || 0) + 1;
      }
    }
  }

  maybeLearnFact(message) {
    const normalized = this.normalize(message);
    const directFact = normalized.match(/^remember that (.+)$/i);
    if (!directFact) {
      return null;
    }

    const fact = directFact[1].trim();
    return this.ingestFact(fact);
  }

  ingestFact(factInput, options = {}) {
    const fact = this.normalize(factInput);
    if (!fact) {
      return { learned: false, reason: 'empty_fact' };
    }

    const source = String(options.source || 'fact_learning');
    const sessionId = String(options.sessionId || 'fact-seed');

    this.store.state.learnedFacts[fact] = {
      count: (this.store.state.learnedFacts[fact]?.count || 0) + 1,
      lastSeen: new Date().toISOString()
    };

    const concepts = this.extractConcepts(fact);
    this.updateConceptGraph(concepts);
    this.updateAssociationGraph(concepts);
    this.updateTokenGraph(this.tokenize(fact));

    this.store.state.interactions.push({
      sessionId,
      user: `remember that ${fact}`,
      bot: `Got it. I will remember this: ${fact}`,
      source,
      confidence: 0.95,
      concepts,
      at: new Date().toISOString()
    });

    if (this.store.state.interactions.length > this.maxMemory) {
      const toTrim = this.store.state.interactions.length - this.maxMemory;
      this.store.state.interactions = this.store.state.interactions.slice(-this.maxMemory);
      this.store.state.trainer.processedUntil = Math.max(0, this.store.state.trainer.processedUntil - toTrim);
    }

    return { learned: true, fact };
  }

  getTopFacts(limit = 2) {
    return Object.entries(this.store.state.learnedFacts)
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, limit)
      .map(([fact]) => fact);
  }

  getRelevantFacts(message, limit = 2) {
    const messageTokens = new Set(this.tokenize(message));
    return Object.keys(this.store.state.learnedFacts)
      .map((fact) => ({
        fact,
        score: this.tokenize(fact).filter((token) => messageTokens.has(token)).length
      }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((item) => item.fact);
  }

  pickWeighted(nextMap) {
    const entries = Object.entries(nextMap || {});
    if (!entries.length) {
      return null;
    }

    const total = entries.reduce((sum, [, weight]) => sum + weight, 0);
    let cursor = Math.random() * total;
    for (const [token, weight] of entries) {
      cursor -= weight;
      if (cursor <= 0) {
        return token;
      }
    }

    return entries[entries.length - 1][0];
  }

  generateMarkovThought(seedTokens = []) {
    const tokenGraph = this.store.state.tokenGraph;
    const words = [];
    let current = seedTokens[0] && tokenGraph[seedTokens[0]] ? seedTokens[0] : '<START>';

    for (let index = 0; index < 22; index += 1) {
      const nextToken = this.pickWeighted(tokenGraph[current]);
      if (!nextToken || nextToken === '<END>') {
        break;
      }

      if (nextToken !== '<START>') {
        words.push(nextToken);
      }

      current = nextToken;
    }

    return words.join(' ').trim();
  }

  scoreSimilarity(a, b) {
    const tokensA = new Set(this.tokenize(a));
    const tokensB = new Set(this.tokenize(b));
    if (!tokensA.size || !tokensB.size) {
      return 0;
    }

    let overlap = 0;
    for (const token of tokensA) {
      if (tokensB.has(token)) {
        overlap += 1;
      }
    }

    return overlap / (tokensA.size + tokensB.size - overlap);
  }

  tokenOverlapCount(a, b) {
    const tokensA = new Set(this.tokenize(a));
    const tokensB = new Set(this.tokenize(b));
    if (!tokensA.size || !tokensB.size) {
      return 0;
    }

    let overlap = 0;
    for (const token of tokensA) {
      if (tokensB.has(token)) {
        overlap += 1;
      }
    }

    return overlap;
  }

  findBestMemory(message, sessionId = null) {
    const interactions = this.store.state.interactions;
    const messageConcepts = this.extractConcepts(message);
    const hasQueryConcepts = messageConcepts.length > 0;
    let best = null;
    let scanned = 0;

    for (let index = interactions.length - 1; index >= 0; index -= 1) {
      scanned += 1;
      if (scanned > this.memorySearchWindow) {
        break;
      }

      const item = interactions[index];
      if (!item || !item.user || !item.bot) {
        continue;
      }

      const sameSession = sessionId && item.sessionId === sessionId;
      if (sessionId && !sameSession && item.source !== 'web_ingest') {
        continue;
      }

      if (
        [
          'memory_match',
          'concept_association',
          'generated',
          'starter_bootstrap',
          'api_fact_ingest',
          'curriculum_fact',
          'fact_learning'
        ].includes(item.source)
      ) {
        continue;
      }

      const normalizedBot = this.normalize(item.bot);
      if (/jump to content\s+main menu/i.test(normalizedBot) || this.isPollutedWebText(item.bot)) {
        continue;
      }

      const itemConcepts = this.extractConcepts(item.user);
      const conceptOverlap = hasQueryConcepts
        ? messageConcepts.filter((concept) => itemConcepts.includes(concept)).length
        : 0;
      if (hasQueryConcepts && conceptOverlap < 1) {
        continue;
      }

      const score = this.scoreSimilarity(message, item.user);
      const overlapCount = this.tokenOverlapCount(message, item.user);

      const minOverlap = item.source === 'web_ingest' ? 1 : 2;
      if (overlapCount < minOverlap) {
        continue;
      }

      if (!sameSession && sessionId) {
        if (score < 0.66 || conceptOverlap < 2 || overlapCount < 3) {
          continue;
        }
      }

      if (!best || score > best.score) {
        best = {
          score,
          item,
          index,
          overlapCount,
          conceptOverlap,
          boostedScore: sameSession ? score + 0.08 : score
        };
      } else if ((sameSession ? score + 0.08 : score) > (best.boostedScore ?? best.score)) {
        best = {
          score,
          item,
          index,
          overlapCount,
          conceptOverlap,
          boostedScore: sameSession ? score + 0.08 : score
        };
      }
    }

    return best && best.score > 0.33 ? best : null;
  }

  buildWebKnowledgeRecallCandidate(message) {
    const normalizedMessage = this.normalize(message);
    const asksDefinition = /(what is|whats|what's|define|who is|tell me about|explain|describe)/.test(normalizedMessage);
    if (!asksDefinition) {
      return null;
    }

    const tokens = this.extractConcepts(message);
    if (!tokens.length) {
      return null;
    }

    const genericTokens = new Set([
      'learn',
      'learning',
      'tell',
      'explain',
      'topic',
      'general',
      'please',
      'about',
      'history',
      'information',
      'details'
    ]);
    const focusTokens = tokens.filter((token) => !genericTokens.has(token));
    const requiredTokens = focusTokens.length ? focusTokens : tokens;

    const entries = Object.entries(this.store.state.webKnowledge || {});
    if (!entries.length) {
      return null;
    }

    const scored = entries
      .map(([url, item]) => {
        const title = String(item?.title || 'Untitled Page');
        const summary = String(item?.lastSummary || '');
        if (this.isPollutedWebText(summary)) {
          return null;
        }
        const normalizedTitle = this.normalize(title);
        const haystack = this.normalize(`${title} ${summary} ${url}`);
        const overlap = requiredTokens.filter((token) => haystack.includes(token)).length;
        const exactTitleHits = requiredTokens.filter((token) => normalizedTitle.includes(token)).length;
        return { url, title, summary, overlap, exactTitleHits };
      })
      .filter((entry) => entry && entry.overlap > 0)
      .sort((left, right) => {
        if (right.exactTitleHits !== left.exactTitleHits) {
          return right.exactTitleHits - left.exactTitleHits;
        }
        return right.overlap - left.overlap;
      });

    if (!scored.length) {
      return null;
    }

    const best = scored[0];
    const minimumOverlap = Math.min(2, requiredTokens.length);
    if (best.overlap < minimumOverlap && best.exactTitleHits < 1) {
      return null;
    }

    const firstSummary = String(best.summary || '')
      .replace(/another related source says\s*:/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 460);

    if (!firstSummary) {
      return null;
    }

    return {
      text: `${firstSummary} (Learned from web source: ${best.title}.)`,
      source: 'web_knowledge_recall',
      baseScore: best.exactTitleHits > 0 ? 0.96 : 0.9
    };
  }

  buildHonestUnknownCandidate(message, concepts) {
    const normalized = this.normalize(message);
    const isSmalltalk = /^(hi|hello|hey|yo|sup|whats up|what's up|how are you|hru|good morning|good afternoon|good evening)$/.test(
      normalized
    );
    if (isSmalltalk) {
      return null;
    }

    const asksDefinition = /(what is|whats|what's|tell me about|about|explain)/.test(normalized);
    if (!asksDefinition) {
      return null;
    }

    const tokenCount = this.tokenize(message).length;
    if (tokenCount <= 2) {
      return null;
    }

    const meaningfulConcepts = concepts.filter((token) => token.length >= 3).slice(0, 3);
    if (!meaningfulConcepts.length) {
      return null;
    }

    const topic = meaningfulConcepts.join(' ');
    return {
      text: `I do not have reliable knowledge about ${topic} yet. Share a URL or a short lesson and I will learn it, then explain it back clearly.`,
      source: 'knowledge_gap',
      baseScore: 0.85
    };
  }

  buildSmallTalkCandidate(message) {
    const normalized = this.normalize(message);
    const tokens = this.tokenize(normalized);
    const greetingTokens = new Set(['hi', 'hello', 'hey', 'yo', 'sup', 'hola']);
    const isRepeatedGreeting =
      tokens.length >= 1 &&
      tokens.length <= 4 &&
      tokens.every((token) => greetingTokens.has(token));

    if (/^(hi|hello|hey|yo|sup|whats up|what's up)$/.test(normalized) || isRepeatedGreeting) {
      return {
        text: 'Hey! I am doing well. Ask me a topic like "what is photosynthesis" or "define algebra" and I will help.',
        source: 'smalltalk',
        baseScore: 0.99
      };
    }

    if (/^(how are you|hru)$/.test(normalized)) {
      return {
        text: 'I am good and ready to help. What do you want to learn right now?',
        source: 'smalltalk',
        baseScore: 0.99
      };
    }

    if (/^(what are you doing|what are you up to|what are u doing|are you okay|are you ok)$/.test(normalized)) {
      return {
        text: 'I am here helping you and learning from this chat. If you want, ask me a topic and I will explain it clearly.',
        source: 'smalltalk',
        baseScore: 0.99
      };
    }

    if (/^what no i ment you as in the ai$/.test(normalized)) {
      return {
        text: 'I am your local AI chat assistant in this project. I can explain topics, answer questions, and learn from your lessons and facts.',
        source: 'smalltalk',
        baseScore: 0.99
      };
    }

    if (/^(whatever|what ever)$/.test(normalized)) {
      return {
        text: 'No worries. If you want to continue, give me any topic and I will keep it short and clear.',
        source: 'smalltalk',
        baseScore: 0.98
      };
    }

    if (/^what the (sigma|heck|hell|frick|fuck|fudge)\b/.test(normalized)) {
      return {
        text: 'I get what you mean. If you want slang help, ask like "what does sigma mean?" or give me a topic to explain.',
        source: 'smalltalk_confusion',
        baseScore: 0.99
      };
    }

    return null;
  }

  inferIntent(message) {
    if (message.includes('?')) {
      return 'question';
    }

    const normalized = this.normalize(message);

    if (/(sad|angry|upset|frustrated|anxious|depressed)/.test(normalized)) {
      return 'emotion';
    }
    if (/(build|create|make|code|project|develop|ship)/.test(normalized)) {
      return 'builder';
    }
    if (/(sentence|grammar|english|write|paragraph|essay)/.test(normalized)) {
      return 'english';
    }
    return 'chat';
  }

  isEssayRequest(message) {
    const normalized = this.normalize(message);
    return /(write|create|generate|make).*(essay|essey|essy|esay)|\b(essay|essey|essy|esay)\b/.test(
      normalized
    );
  }

  clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  pickByTemperature(items, temperature) {
    if (!Array.isArray(items) || !items.length) {
      return null;
    }

    const t = this.clamp(Number(temperature) || 0, 0, 1);
    if (t <= 0.12) {
      return items[0];
    }

    const randomIndex = Math.floor(Math.random() * items.length);
    const anchoredIndex = Math.floor((items.length - 1) * (1 - t) * 0.3);
    return Math.random() < 0.22 ? items[anchoredIndex] : items[randomIndex];
  }

  resolveEssayTemperature(message) {
    const normalized = this.normalize(message);
    const explicitMatch = normalized.match(/temperature\s*(\d(?:\.\d+)?)/i);
    if (explicitMatch) {
      return this.clamp(Number(explicitMatch[1]), 0, 1);
    }

    if (/(creative|imaginative|fun|vivid|poetic)/.test(normalized)) {
      return this.clamp(this.essayTemperature + 0.2, 0, 1);
    }
    if (/(formal|academic|strict|concise)/.test(normalized)) {
      return this.clamp(this.essayTemperature - 0.2, 0, 1);
    }

    return this.essayTemperature;
  }

  splitIntoSentences(text) {
    return String(text || '')
      .split(/(?<=[.!?])\s+/)
      .map((line) => line.trim())
      .filter((line) => line.length >= 40 && line.length <= 260);
  }

  getTopicEvidence(topic, limit = 8) {
    const topicTokens = this.tokenize(topic);
    if (!topicTokens.length) {
      return [];
    }

    const scored = [];
    const interactions = this.store.state.interactions;
    for (let index = interactions.length - 1; index >= 0; index -= 1) {
      const item = interactions[index];
      if (!item || !item.bot) {
        continue;
      }
      if (item.source === 'starter_bootstrap') {
        continue;
      }

      const overlap = topicTokens.filter((token) => this.normalize(`${item.user} ${item.bot}`).includes(token)).length;
      if (overlap <= 0) {
        continue;
      }

      const sentences = this.splitIntoSentences(item.bot);
      for (const sentence of sentences) {
        const sentenceOverlap = topicTokens.filter((token) => this.normalize(sentence).includes(token)).length;
        if (sentenceOverlap <= 0) {
          continue;
        }
        scored.push({
          sentence,
          score: sentenceOverlap * 2 + (item.confidence || 0)
        });
      }
    }

    return scored
      .sort((left, right) => right.score - left.score)
      .map((entry) => entry.sentence)
      .filter((value, idx, arr) => arr.indexOf(value) === idx)
      .slice(0, limit);
  }

  buildBrainEssayParagraph(topic, evidencePool, temperature, purpose) {
    const selectedEvidence = this.pickByTemperature(evidencePool, temperature);
    const thought = this.generateMarkovThought(this.tokenize(`${topic} ${purpose}`));
    const cleanThought = this.isLowQualityThought(thought)
      ? `This part of the essay focuses on ${topic} from a practical and reasoning-based perspective.`
      : thought[0].toUpperCase() + thought.slice(1);

    if (selectedEvidence) {
      const leadIns = [
        `A useful idea from prior learning is that ${selectedEvidence}`,
        `From memory, one strong point is: ${selectedEvidence}`,
        `Evidence from earlier conversations suggests that ${selectedEvidence}`
      ];
      return `${this.pickByTemperature(leadIns, temperature)} ${cleanThought}`;
    }

    return cleanThought;
  }

  extractEssayTopic(message) {
    const normalized = this.normalize(message);

    const aboutMatch = normalized.match(/(essay|essey|essy|esay)\s+(about|on|regarding)\s+([a-z0-9\s-]{2,120})/i);
    if (aboutMatch && aboutMatch[3]) {
      return aboutMatch[3].trim();
    }

    const trailingMatch = normalized.match(/(about|on|regarding)\s+([a-z0-9\s-]{2,120})$/i);
    if (trailingMatch && trailingMatch[2]) {
      return trailingMatch[2].trim();
    }

    if (/random\s+essay/.test(normalized)) {
      return 'continuous learning and growth mindset';
    }

    return 'effective communication and learning';
  }

  buildEssayCandidate(message) {
    if (!this.isEssayRequest(message)) {
      return null;
    }

    const topic = this.extractEssayTopic(message);
    const temperature = this.resolveEssayTemperature(message);
    const evidence = this.getTopicEvidence(topic, 10);
    const title = topic
      .split(' ')
      .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : word))
      .join(' ');

    const isPluralTopic = /s$/.test(topic.trim());
    const beVerb = isPluralTopic ? 'are' : 'is';
    const opener = this.pickByTemperature(
      [
        `${title} ${beVerb} worth studying because it influences how we reason and make decisions.`,
        `An essay about ${topic} should combine evidence, interpretation, and practical implications.`,
        `To understand ${topic}, we should connect learned knowledge with clear explanation.`
      ],
      temperature
    );

    const paragraph1 = `${opener} ${this.buildBrainEssayParagraph(topic, evidence, temperature, 'introduction')}`;
    const paragraph2 = this.buildBrainEssayParagraph(topic, evidence, temperature, 'analysis');
    const paragraph3 = `${this.buildBrainEssayParagraph(topic, evidence, temperature, 'conclusion')} In conclusion, ${topic} ${beVerb} important for clear thinking, communication, and better decisions.`;

    const maybeBridge = temperature >= 0.72
      ? `

A key lesson here is that understanding improves when we compare multiple viewpoints and update beliefs based on stronger evidence.`
      : '';

    return {
      text: `${title}\n\n${paragraph1}\n\n${paragraph2}${maybeBridge}\n\n${paragraph3}`,
      source: 'essay_brain_writer',
      baseScore: 0.96
    };
  }

  getRequestedSentenceCount(message) {
    const normalized = this.normalize(message);
    const numberMatch = normalized.match(/\b(\d{1,2})\b/);
    if (!numberMatch) {
      return 3;
    }

    const count = Number(numberMatch[1]);
    if (!Number.isFinite(count) || count <= 0) {
      return 3;
    }

    return Math.min(8, count);
  }

  extractSentenceTopic(message) {
    const raw = String(message || '').trim();
    if (!raw) {
      return 'daily life';
    }

    const normalized = this.normalize(raw);
    const topicMatch = normalized.match(
      /(about|on|regarding|for)\s+([a-z0-9\s-]{2,80})$/i
    );
    if (topicMatch && topicMatch[2]) {
      return topicMatch[2].trim();
    }

    const cleaned = normalized
      .replace(/\b(make|create|write|give|generate|build|english|sentence|sentences|example|examples|please)\b/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    return cleaned || 'daily life';
  }

  isSentenceCreationRequest(message) {
    const normalized = this.normalize(message);
    const asksForSentences = /(make|create|write|generate|give).*(sentence|sentences|example|examples)/.test(
      normalized
    );
    const directSentenceAsk = /(sentence|sentences)\s+(about|for|on)/.test(normalized);
    return asksForSentences || directSentenceAsk;
  }

  generateEnglishSentences(topic, count) {
    const safeTopic = topic.trim() || 'daily life';
    const templates = [
      `The topic of ${safeTopic} is important in everyday communication.`,
      `I am learning how to explain ${safeTopic} in clear English sentences.`,
      `Many students improve quickly when they practice writing about ${safeTopic}.`,
      `A simple sentence about ${safeTopic} can still be precise and meaningful.`,
      `To understand ${safeTopic}, it helps to read and write regularly.`,
      `Clear grammar makes ideas about ${safeTopic} easier to follow.`,
      `You can build confidence by speaking about ${safeTopic} step by step.`,
      `Short, accurate sentences about ${safeTopic} are often best for beginners.`
    ];

    return templates.slice(0, Math.max(1, Math.min(count, templates.length)));
  }

  buildEnglishSentenceCandidate(message) {
    if (!this.isSentenceCreationRequest(message)) {
      return null;
    }

    const count = this.getRequestedSentenceCount(message);
    const topic = this.extractSentenceTopic(message);
    const sentences = this.generateEnglishSentences(topic, count);

    return {
      text: `Here are ${sentences.length} English sentence example(s) about ${topic}: ${sentences.join(' ')}`,
      source: 'english_sentence_builder',
      baseScore: 0.95
    };
  }

  isLikelyLessonMessage(message) {
    const source = String(message || '').trim();
    if (!source) {
      return false;
    }

    const sentenceCount = (source.match(/[.!?](\s|$)/g) || []).length;
    const length = source.length;
    const teachingCues = /(grammar|english|math|algebra|tense|sentence|subject|verb|noun|equation|conditional|punctuation)/i.test(
      source
    );

    return length >= 280 && sentenceCount >= 5 && teachingCues;
  }

  buildLessonCandidate(message, concepts) {
    if (!this.isLikelyLessonMessage(message)) {
      return null;
    }

    const topConcepts = concepts.slice(0, 8);
    const sentenceCount = (String(message).match(/[.!?](\s|$)/g) || []).length;

    return {
      text: `Thanks, this is a strong teaching lesson. I learned ${sentenceCount} key statements and extracted concepts like: ${topConcepts.join(', ')}. Ask me to quiz you, summarize this, or apply these rules to examples.`,
      source: 'lesson_ingest',
      baseScore: 0.9
    };
  }

  isLowQualityThought(text) {
    const sample = String(text || '').trim();
    if (!sample) {
      return true;
    }

    const words = sample.split(/\s+/).filter(Boolean);
    if (words.length < 6) {
      return true;
    }

    const weirdTokenCount = words.filter((word) => /\d{3,}/.test(word) || word.length > 22).length;
    return weirdTokenCount / words.length > 0.25;
  }

  isLowSignalInput(message) {
    const raw = String(message || '').trim();
    if (!raw) {
      return true;
    }

    const tokens = this.tokenize(raw);
    if (!tokens.length) {
      return true;
    }

    const normalized = this.normalize(raw);
    const singleTokenAmbiguous = new Set([
      'what',
      'huh',
      'wut',
      'uh',
      'um',
      'hmm',
      'hmmm',
      'why',
      'how'
    ]);

    if (tokens.length === 1 && singleTokenAmbiguous.has(tokens[0])) {
      return true;
    }

    if (/^what the (sigma|heck|hell|frick|fuck|fudge)\b/.test(normalized)) {
      return true;
    }

    if (tokens.length <= 3 && /(fuck|frick|wtf|faj)/.test(normalized)) {
      return true;
    }

    if (tokens.length === 1) {
      const token = tokens[0];
      const consonantRuns = token.match(/[^aeiou]{4,}/g) || [];
      const knownConcept = Boolean(this.store.state.conceptGraph?.[token]);
      const vowelCount = (token.match(/[aeiou]/g) || []).length;

      if (consonantRuns.length > 0) {
        return true;
      }
      if (token.length >= 6 && vowelCount <= 1) {
        return true;
      }
      if (token.length >= 8 && !knownConcept) {
        return true;
      }
    }

    const meaningfulTokens = tokens.filter((token) => /[aeiou]/.test(token) && token.length >= 3);
    const keyboardMashes = /(asdf|qwer|zxcv|hjkl|poiuy|mnbv)/i.test(raw.replace(/\s+/g, ''));

    if (keyboardMashes) {
      return true;
    }

    if (tokens.length <= 2 && meaningfulTokens.length === 0) {
      return true;
    }

    const mostlySymbols = raw.replace(/[a-z0-9\s]/gi, '').length > raw.length * 0.45;
    if (mostlySymbols) {
      return true;
    }

    const looksLikeDateOnly = /\b(january|february|march|april|may|june|july|august|september|october|november|december)\b/i.test(
      raw
    ) && /\b\d{1,2}\b/.test(raw) && /\b\d{4}\b/.test(raw);
    const hasQuestionCue = /(what|why|how|when|who|explain|define|tell me)/i.test(raw);
    if (looksLikeDateOnly && !hasQuestionCue) {
      return true;
    }

    return false;
  }

  buildClarificationCandidate(message) {
    if (!this.isLowSignalInput(message)) {
      return null;
    }

    const normalized = this.normalize(message);
    if (/^(hi|hello|hey|yo|sup|hola)$/.test(normalized)) {
      return {
        text: 'Hey. I am here and ready. Ask a specific question like "what is a hat" or "explain algebra".',
        source: 'smalltalk_greeting',
        baseScore: 0.98
      };
    }

    if (/^(what|huh|wut|eh|um|uh)$/.test(normalized)) {
      return {
        text: 'I might have missed your intent. Tell me the topic in one line, for example: "define cat", "what is gravity", or "explain tenses".',
        source: 'smalltalk_confusion',
        baseScore: 0.98
      };
    }

    if (/(fuck|frick|wtf|faj)/.test(normalized)) {
      return {
        text: 'I get the frustration. Give me the exact question or topic and I will answer directly without repeating old context.',
        source: 'smalltalk_frustration',
        baseScore: 0.98
      };
    }

    return {
      text: 'I could not understand that input clearly yet. Please ask a full question or share a topic (for example: "Explain English tenses" or "Solve 2x+4=10").',
      source: 'clarification_needed',
      baseScore: 0.97
    };
  }

  formatNumber(value) {
    if (!Number.isFinite(value)) {
      return 'undefined';
    }

    const rounded = Number(value.toFixed(8));
    return Number.isInteger(rounded) ? String(rounded) : String(rounded);
  }

  isLikelyMathMessage(message) {
    const sample = String(message || '').trim();
    if (!sample) {
      return false;
    }

    const hasOperator = /[+\-*/^=]/.test(sample);
    const hasDigit = /\d/.test(sample);
    const mathCue = /(solve|calculate|evaluate|what is|math|equation|simplify)/i.test(sample);

    return (hasOperator && hasDigit) || (mathCue && hasDigit);
  }

  buildIdentityCandidate(message) {
    const normalized = this.normalize(message);
    const asksModel = /(what|which).*(model|ai model)|your ai model|what are you|who are you/.test(normalized);
    if (!asksModel) {
      return null;
    }

    return {
      text: 'I am your local learning student bot. I generate the main reply from my own memory and training. A separate mentor AI may review and improve quality, but I am still the one speaking to you.',
      source: 'identity_explain',
      baseScore: 0.88
    };
  }

  tokenizeMathExpression(expression) {
    const sanitized = String(expression || '').replace(/\s+/g, '');
    if (!sanitized) {
      return [];
    }

    const tokens = [];
    let index = 0;

    while (index < sanitized.length) {
      const char = sanitized[index];

      if (/\d|\./.test(char) || ((char === '+' || char === '-') && (index === 0 || /[+\-*/^(]/.test(sanitized[index - 1])))) {
        let numberText = char;
        index += 1;
        while (index < sanitized.length && /\d|\./.test(sanitized[index])) {
          numberText += sanitized[index];
          index += 1;
        }

        if (!/^[-+]?\d*\.?\d+$/.test(numberText)) {
          throw new Error('invalid_number');
        }

        tokens.push({ type: 'number', value: Number(numberText) });
        continue;
      }

      if (MATH_PRECEDENCE[char]) {
        tokens.push({ type: 'operator', value: char });
        index += 1;
        continue;
      }

      if (char === '(' || char === ')') {
        tokens.push({ type: 'paren', value: char });
        index += 1;
        continue;
      }

      throw new Error('invalid_character');
    }

    return tokens;
  }

  toRpn(tokens) {
    const output = [];
    const operators = [];

    for (const token of tokens) {
      if (token.type === 'number') {
        output.push(token);
      } else if (token.type === 'operator') {
        while (operators.length) {
          const top = operators[operators.length - 1];
          if (top.type !== 'operator') {
            break;
          }

          const topPrec = MATH_PRECEDENCE[top.value];
          const currentPrec = MATH_PRECEDENCE[token.value];
          const shouldPop = RIGHT_ASSOCIATIVE.has(token.value)
            ? currentPrec < topPrec
            : currentPrec <= topPrec;

          if (!shouldPop) {
            break;
          }

          output.push(operators.pop());
        }
        operators.push(token);
      } else if (token.type === 'paren' && token.value === '(') {
        operators.push(token);
      } else if (token.type === 'paren' && token.value === ')') {
        while (operators.length && operators[operators.length - 1].value !== '(') {
          output.push(operators.pop());
        }
        if (!operators.length || operators[operators.length - 1].value !== '(') {
          throw new Error('mismatched_parentheses');
        }
        operators.pop();
      }
    }

    while (operators.length) {
      const top = operators.pop();
      if (top.type === 'paren') {
        throw new Error('mismatched_parentheses');
      }
      output.push(top);
    }

    return output;
  }

  evaluateRpn(rpnTokens) {
    const stack = [];

    for (const token of rpnTokens) {
      if (token.type === 'number') {
        stack.push(token.value);
        continue;
      }

      const right = stack.pop();
      const left = stack.pop();
      if (left === undefined || right === undefined) {
        throw new Error('invalid_expression');
      }

      let result;
      if (token.value === '+') {
        result = left + right;
      } else if (token.value === '-') {
        result = left - right;
      } else if (token.value === '*') {
        result = left * right;
      } else if (token.value === '/') {
        if (right === 0) {
          throw new Error('division_by_zero');
        }
        result = left / right;
      } else if (token.value === '^') {
        result = left ** right;
      }

      if (!Number.isFinite(result) || Math.abs(result) > 1e12) {
        throw new Error('result_out_of_range');
      }

      stack.push(result);
    }

    if (stack.length !== 1) {
      throw new Error('invalid_expression');
    }

    return stack[0];
  }

  extractArithmeticExpression(message) {
    const source = String(message || '');
    const match = source.match(/[0-9+\-*/^().\s]{3,}/g);
    if (!match || !match.length) {
      return null;
    }

    const expression = match.sort((a, b) => b.length - a.length)[0].trim();
    if (!/[+\-*/^]/.test(expression) || !/\d/.test(expression)) {
      return null;
    }

    return expression;
  }

  solveLinearEquation(message) {
    const compact = String(message || '')
      .toLowerCase()
      .replace(/\s+/g, '')
      .replace(/\?+$/, '');

    const equationMatch = compact.match(/([+-]?\d*\.?\d*x(?:[+-]\d*\.?\d+)?=[+-]?\d*\.?\d+)/);
    if (!equationMatch) {
      return null;
    }

    const equation = equationMatch[1];
    const match = equation.match(/^([+-]?\d*\.?\d*)x([+-]\d*\.?\d+)?=([+-]?\d*\.?\d+)$/);
    if (!match) {
      return null;
    }

    const aRaw = match[1];
    const bRaw = match[2] || '0';
    const cRaw = match[3];

    const a = aRaw === '' || aRaw === '+' ? 1 : aRaw === '-' ? -1 : Number(aRaw);
    const b = Number(bRaw);
    const c = Number(cRaw);

    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || a === 0) {
      return null;
    }

    const x = (c - b) / a;
    if (!Number.isFinite(x)) {
      return null;
    }

    return {
      expression: equation,
      a,
      b,
      c,
      result: x
    };
  }

  buildMathCandidate(message) {
    if (!this.isLikelyMathMessage(message)) {
      return null;
    }

    const linear = this.solveLinearEquation(message);
    if (linear) {
      const a = linear.a;
      const b = linear.b;
      const c = linear.c;
      const step1 = `${a}x ${b >= 0 ? '+' : '-'} ${Math.abs(b)} = ${c}`;
      const step2 = `${a}x = ${c - b}`;
      const step3 = `x = ${this.formatNumber((c - b) / a)}`;

      return {
        text: `Math result: for ${linear.expression}, x = ${this.formatNumber(linear.result)}.`,
        source: 'math_solver',
        baseScore: 0.84,
        mathMeta: {
          kind: 'linear_equation',
          expression: linear.expression,
          result: linear.result,
          steps: [step1, step2, step3]
        }
      };
    }

    const arithmetic = this.extractArithmeticExpression(message);
    if (!arithmetic) {
      return null;
    }

    try {
      const tokens = this.tokenizeMathExpression(arithmetic);
      const rpn = this.toRpn(tokens);
      const value = this.evaluateRpn(rpn);
      return {
        text: `Math result: ${arithmetic} = ${this.formatNumber(value)}.`,
        source: 'math_solver',
        baseScore: 0.82,
        mathMeta: {
          kind: 'arithmetic',
          expression: arithmetic,
          result: value,
          steps: [`Evaluate expression using order of operations: ${arithmetic} = ${this.formatNumber(value)}`]
        }
      };
    } catch (error) {
      return {
        text: `I detected a math expression (${arithmetic}) but could not solve it safely (${error.message}). Try a simpler arithmetic form like (2+5)*3 or an equation like 2x+4=10.`,
        source: 'math_solver',
        baseScore: 0.52
      };
    }
  }

  getRecentSessionMathMemory(sessionId) {
    if (!sessionId) {
      return null;
    }

    for (let index = this.store.state.interactions.length - 1; index >= 0; index -= 1) {
      const interaction = this.store.state.interactions[index];
      if (interaction.sessionId !== sessionId) {
        continue;
      }
      if (interaction.source === 'math_solver' && interaction.mathMeta) {
        return interaction;
      }
    }

    return null;
  }

  buildMathFollowupCandidate(message, sessionId) {
    const normalized = this.normalize(message);
    const asksWhy = /(why|how|explain|show steps|what happened|how come|equal to)/.test(normalized);
    if (!asksWhy) {
      return null;
    }

    const recentMath = this.getRecentSessionMathMemory(sessionId);
    if (!recentMath || !recentMath.mathMeta) {
      return null;
    }

    const meta = recentMath.mathMeta;
    const steps = Array.isArray(meta.steps) ? meta.steps : [];
    const stepText = steps.length ? steps.map((step, idx) => `${idx + 1}) ${step}`).join(' ') : 'I applied standard math steps to isolate and evaluate the expression.';

    return {
      text: `Great follow-up. For ${meta.expression}, here is why: ${stepText} So the answer is ${this.formatNumber(meta.result)}.`,
      source: 'math_explain',
      baseScore: 0.93,
      mathMeta: meta
    };
  }

  getAssociatedConcepts(messageConcepts, limit = 3) {
    const associationGraph = this.store.state.associationGraph;
    const scored = {};

    for (const concept of messageConcepts) {
      const neighbors = associationGraph[concept] || {};
      for (const [candidate, score] of Object.entries(neighbors)) {
        if (messageConcepts.includes(candidate)) {
          continue;
        }
        scored[candidate] = (scored[candidate] || 0) + score;
      }
    }

    return Object.entries(scored)
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([concept]) => concept);
  }

  scoreCandidate(candidate, message, concepts) {
    let score = candidate.baseScore;
    if (candidate.source === 'response_bank') {
      score += 0.15;
    }

    const overlapScore = this.scoreSimilarity(message, candidate.text);
    score += overlapScore * 0.35;

    const conceptHits = concepts.filter((concept) => this.normalize(candidate.text).includes(concept)).length;
    score += Math.min(0.25, conceptHits * 0.06);

    const normalizedMessage = this.normalize(message);
    const asksDefinition = /(what is|whats|what's|tell me about|explain|define)/.test(normalizedMessage);
    if (asksDefinition && candidate.source === 'concept_association') {
      score -= 0.28;
    }
    if (asksDefinition && candidate.source === 'generated') {
      score -= 0.18;
    }
    if (asksDefinition && candidate.source === 'memory_match' && overlapScore < 0.4) {
      score -= 0.2;
    }
    if (candidate.source === 'neural_memory') {
      score += 0.08;
    }

    return Math.min(0.99, Math.max(0.01, score));
  }

  buildCandidates(message, concepts, sessionId) {
    const candidates = [];
    const normalized = this.normalize(message);

    const smallTalkCandidate = this.buildSmallTalkCandidate(message);
    if (smallTalkCandidate) {
      candidates.push(smallTalkCandidate);
    }

    const webKnowledgeRecall = this.buildWebKnowledgeRecallCandidate(message);
    if (webKnowledgeRecall) {
      candidates.push(webKnowledgeRecall);
    }

    const honestUnknownCandidate = this.buildHonestUnknownCandidate(message, concepts);
    if (honestUnknownCandidate) {
      candidates.push(honestUnknownCandidate);
    }

    const essayCandidate = this.buildEssayCandidate(message);
    if (essayCandidate) {
      candidates.push(essayCandidate);
    }

    const englishSentenceCandidate = this.buildEnglishSentenceCandidate(message);
    if (englishSentenceCandidate) {
      candidates.push(englishSentenceCandidate);
    }

    const mathFollowupCandidate = this.buildMathFollowupCandidate(message, sessionId);
    if (mathFollowupCandidate) {
      candidates.push(mathFollowupCandidate);
    }

    const clarificationCandidate = this.buildClarificationCandidate(message);
    if (clarificationCandidate) {
      candidates.push(clarificationCandidate);
      return candidates;
    }

    const lessonCandidate = this.buildLessonCandidate(message, concepts);
    if (lessonCandidate) {
      candidates.push(lessonCandidate);
    }

    const identityCandidate = this.buildIdentityCandidate(message);
    if (identityCandidate) {
      candidates.push(identityCandidate);
    }

    const mathCandidate = this.buildMathCandidate(message);
    if (mathCandidate) {
      candidates.push(mathCandidate);
    }

    const direct = this.store.state.responseBank[normalized];
    if (direct && direct.length) {
      candidates.push({
        text: direct[0].reply,
        source: 'response_bank',
        baseScore: 0.62
      });
    }

    const recalled = this.findBestMemory(message, sessionId);
    if (recalled) {
      const cleanedMemoryReply = String(recalled.item.bot)
        .replace(/\s*\(I used a similar memory from past chats\.\)\s*/gi, ' ')
        .replace(/\s+/g, ' ')
        .trim();

      candidates.push({
        text: `${cleanedMemoryReply} (I used a similar memory from past chats.)`,
        source: 'memory_match',
        baseScore: 0.5 + recalled.score * 0.3,
        memoryRef: {
          interactionIndex: recalled.index,
          at: recalled.item.at,
          user: recalled.item.user
        }
      });
    }

    const neuralCandidate = this.buildNeuralCandidate(message);
    if (neuralCandidate) {
      candidates.push(neuralCandidate);
    }

    const related = concepts.length >= 2 ? this.getAssociatedConcepts(concepts, 3) : [];
    if (related.length) {
      // Only offer explicit concept-association suggestions when the user asks
      // for related topics or connections. Avoid auto-injecting this on every
      // short/ambiguous input which felt noisy during casual chat.
      const wantsAssociations = /(connect|related to|related|associate|associat|similar to|what about|connections|related topics|also)/.test(
        this.normalize(message)
      );
      if (wantsAssociations) {
        candidates.push({
          text: `I think this relates to: ${related.join(', ')}. Tell me if you want me to expand on any of those.`,
          source: 'concept_association',
          baseScore: 0.55
        });
      }
    }

    const thought = this.generateMarkovThought(this.tokenize(message));
    const qualityThought = this.isLowQualityThought(thought) ? '' : thought;
    const facts = this.getRelevantFacts(message, 2);
    const intent = this.inferIntent(message);

    let opener = 'I am continuously training from conversations, and my current reasoning is:';
    if (intent === 'question') {
      opener = 'Great question. I am still training, and my current reasoning is:';
    } else if (intent === 'builder') {
      opener = 'I can help you build this. My learned pattern suggests:';
    } else if (intent === 'emotion') {
      opener = 'I hear you. I am learning to respond better, and my current thought is:';
    }

    const insight = qualityThought || 'I understood the topic and stored it in memory for better future answers.';
    const factLine = facts.length ? ` I currently remember: ${facts.join('; ')}.` : '';

    candidates.push({
      text: `${opener} ${insight}.${factLine} Share another example so I can improve.`,
      source: 'generated',
      baseScore: qualityThought ? 0.48 : 0.43
    });

    return candidates;
  }

  buildWebContextCandidate(webContexts = []) {
    if (!webContexts.length) {
      return null;
    }

    const sourceSummaries = webContexts
      .slice(0, 2)
      .map((item) => {
        const title = String(item.title || 'Web source').trim();
        const snippet = String(item.text || '')
          .replace(/\s+/g, ' ')
          .trim()
          .slice(0, 220);
        return `From ${title}: ${snippet}`;
      })
      .join(' ');

    if (!sourceSummaries.trim()) {
      return null;
    }

    return {
      text: `${sourceSummaries}. I can use more links like this to keep improving my understanding.`,
      source: 'web_context',
      baseScore: 0.66
    };
  }

  chooseBestCandidate(message, concepts, webContexts = [], sessionId) {
    const candidates = this.buildCandidates(message, concepts, sessionId);
    const webCandidate = this.buildWebContextCandidate(webContexts);
    if (webCandidate) {
      candidates.unshift(webCandidate);
    }

    const scored = candidates.map((candidate) => ({
      ...candidate,
      confidence: this.scoreCandidate(candidate, message, concepts)
    }));

    const session = this.store.state.sessions[sessionId];
    const lastTurn =
      session && Array.isArray(session.recentTurns) && session.recentTurns.length
        ? session.recentTurns[session.recentTurns.length - 1]
        : null;

    if (lastTurn && lastTurn.bot) {
      const userContinuity = this.scoreSimilarity(message, lastTurn.user || '');
      const followupSignal = /(it|that|this|they|them|those|these|same|again|continue|more|why|how)/.test(
        this.normalize(message)
      );

      for (const item of scored) {
        if (['clarification_needed', 'knowledge_gap'].includes(item.source)) {
          continue;
        }

        const repeatScore = this.scoreSimilarity(item.text, lastTurn.bot);
        if (repeatScore >= 0.86 && userContinuity < 0.25 && !followupSignal) {
          item.confidence = Math.max(0.01, item.confidence - 0.42);
        }
      }
    }

    scored.sort((left, right) => right.confidence - left.confidence);

    const normalizedMessage = this.normalize(message);
    const asksDefinition = /(what is|whats|what's|tell me about|explain|define)/.test(normalizedMessage);
    if (asksDefinition && scored[0]?.source === 'concept_association') {
      const gap = scored.find((item) => item.source === 'knowledge_gap');
      if (gap) {
        return gap;
      }
    }

    return scored[0];
  }

  ingestWebsiteKnowledge({ url, title, text, sessionId }) {
    const normalizedText = String(text || '').trim();
    if (!normalizedText) {
      return;
    }

    if (this.isPollutedWebText(normalizedText)) {
      return;
    }

    const knowledge = this.store.state.webKnowledge;
    if (!knowledge[url]) {
      knowledge[url] = {
        title: title || 'Untitled Page',
        fetchCount: 0,
        learnedChars: 0,
        lastSeen: null,
        lastSummary: ''
      };
    }

    knowledge[url].title = title || knowledge[url].title;
    knowledge[url].fetchCount += 1;
    knowledge[url].learnedChars += normalizedText.length;
    knowledge[url].lastSeen = new Date().toISOString();
    knowledge[url].lastSummary = normalizedText.slice(0, 600);

    const concepts = this.extractConcepts(normalizedText);
    this.updateConceptGraph(concepts);
    this.updateAssociationGraph(concepts);
    this.updateTokenGraph(this.tokenize(normalizedText));

    this.store.state.stats.webIngestions += 1;

    this.store.state.interactions.push({
      sessionId,
      user: `[WEB:${url}] ${title || 'Untitled Page'}`,
      bot: normalizedText.slice(0, 280),
      source: 'web_ingest',
      confidence: 0.86,
      concepts,
      at: new Date().toISOString()
    });
  }

  ingestStarterLesson(lesson) {
    if (!lesson || typeof lesson !== 'object') {
      return { loaded: false, reason: 'invalid_lesson' };
    }

    const id = String(lesson.id || '').trim();
    const topic = String(lesson.topic || '').trim() || 'general';
    const content = String(lesson.content || '').trim();

    if (!id || !content) {
      return { loaded: false, reason: 'missing_id_or_content' };
    }

    if (this.store.state.starterLessons[id]) {
      return { loaded: false, reason: 'already_loaded' };
    }

    const concepts = this.extractConcepts(content);
    this.updateConceptGraph(concepts);
    this.updateAssociationGraph(concepts);
    this.updateTokenGraph(this.tokenize(content));

    this.store.state.starterLessons[id] = {
      topic,
      loadedAt: new Date().toISOString(),
      chars: content.length
    };

    this.store.state.stats.starterLessonsLoaded += 1;
    this.store.state.interactions.push({
      sessionId: 'starter-seed',
      user: `[STARTER:${topic}] ${id}`,
      bot: content.slice(0, 300),
      source: 'starter_bootstrap',
      confidence: 0.9,
      concepts,
      at: new Date().toISOString()
    });

    if (this.store.state.interactions.length > this.maxMemory) {
      const toTrim = this.store.state.interactions.length - this.maxMemory;
      this.store.state.interactions = this.store.state.interactions.slice(-this.maxMemory);
      this.store.state.trainer.processedUntil = Math.max(0, this.store.state.trainer.processedUntil - toTrim);
    }

    return { loaded: true, id };
  }

  reinforceWithMentor({ sessionId, message, finalReply, feedback }) {
    if (!finalReply) {
      return;
    }

    this.rememberResponse(message, finalReply);
    this.store.state.stats.mentorGuidances += 1;

    this.store.state.interactions.push({
      sessionId,
      user: `${message} [MENTOR]`,
      bot: finalReply,
      source: 'mentor_guided',
      confidence: 0.93,
      concepts: this.extractConcepts(`${message} ${feedback || ''}`),
      at: new Date().toISOString()
    });
  }

  replaceIncorrectMemory({ memoryRef, message, badReply, correctedReply }) {
    if (!memoryRef || typeof memoryRef.interactionIndex !== 'number') {
      return { removed: false, reason: 'missing_memory_ref' };
    }

    const interactions = this.store.state.interactions;
    const index = memoryRef.interactionIndex;

    if (index < 0 || index >= interactions.length) {
      return { removed: false, reason: 'memory_index_out_of_range' };
    }

    const candidate = interactions[index];
    const looksLikeSameMemory =
      candidate &&
      candidate.at === memoryRef.at &&
      String(candidate.user || '').toLowerCase() === String(memoryRef.user || '').toLowerCase();

    if (!looksLikeSameMemory) {
      return { removed: false, reason: 'memory_reference_mismatch' };
    }

    interactions.splice(index, 1);
    this.store.state.trainer.processedUntil = Math.max(0, this.store.state.trainer.processedUntil - 1);

    const key = this.normalize(message || memoryRef.user || '');
    if (key && this.store.state.responseBank[key]) {
      this.store.state.responseBank[key] = this.store.state.responseBank[key].filter(
        (entry) => entry.reply !== badReply
      );

      if (!this.store.state.responseBank[key].length) {
        delete this.store.state.responseBank[key];
      }
    }

    if (key && correctedReply) {
      this.rememberResponse(message || memoryRef.user, correctedReply);
    }

    return { removed: true };
  }

  learnFromInteraction(interaction) {
    if (!interaction || !interaction.user) {
      return;
    }

    const userTokens = this.tokenize(interaction.user);
    this.updateTokenGraph(userTokens);

    const concepts = this.extractConcepts(interaction.user);
    this.updateConceptGraph(concepts);
    this.updateAssociationGraph(concepts);

    if (interaction.bot) {
      this.rememberResponse(interaction.user, interaction.bot);
    }

    this.trainNeuralFromInteraction(interaction);
  }

  runTrainerTick(batchSize = 30) {
    const trainer = this.store.state.trainer;
    const interactions = this.store.state.interactions;

    const startIndex = trainer.processedUntil || 0;
    if (startIndex >= interactions.length) {
      trainer.lastRunAt = new Date().toISOString();
      this.store.state.stats.trainerIterations += 1;
      return { processed: 0, remaining: 0 };
    }

    const endIndex = Math.min(startIndex + batchSize, interactions.length);
    for (let index = startIndex; index < endIndex; index += 1) {
      this.learnFromInteraction(interactions[index]);
    }

    const processed = endIndex - startIndex;
    trainer.processedUntil = endIndex;
    trainer.lastRunAt = new Date().toISOString();
    this.store.state.stats.trainerIterations += 1;
    this.store.state.stats.trainerProcessedInteractions += processed;
    if (processed > 0) {
      this.store.scheduleSave(['core', 'language', 'neural']);
    }

    const remaining = interactions.length - endIndex;
    debug('Trainer tick processed=%d remaining=%d', processed, remaining);

    return { processed, remaining };
  }

  ingestExternalReply({
    sessionId,
    message,
    reply,
    source = 'neural_primary',
    confidence = 0.72,
    mathMeta = null,
    webContexts = []
  }) {
    this.ensureSession(sessionId);
    this.maybeLearnFact(message);

    const contextText = webContexts.map((item) => item.text || '').join(' ');
    const concepts = this.extractConcepts(`${message} ${contextText}`);

    this.updateTokenGraph(this.tokenize(message));
    this.updateConceptGraph(concepts);
    this.updateAssociationGraph(concepts);
    this.rememberResponse(message, reply);

    this.store.state.interactions.push({
      sessionId,
      user: message,
      bot: reply,
      source,
      confidence,
      mathMeta,
      concepts,
      at: new Date().toISOString()
    });

    if (this.store.state.interactions.length > this.maxMemory) {
      const toTrim = this.store.state.interactions.length - this.maxMemory;
      this.store.state.interactions = this.store.state.interactions.slice(-this.maxMemory);
      this.store.state.trainer.processedUntil = Math.max(0, this.store.state.trainer.processedUntil - toTrim);
    }

    this.store.state.stats.messages += 1;
    this.store.state.sessions[sessionId].turns += 1;
    this.rememberSessionTurn(sessionId, message, reply, source);

    const externalSession = this.store.state.sessions[sessionId];
    if (externalSession) {
      const asksClarification = /(could you please rephrase|could you rephrase|can you clarify|please clarify|i don'?t understand|do not understand)/i.test(
        String(reply || '')
      );
      if (asksClarification) {
        const anchorTurn = this.getLastContextAnchorTurn(sessionId);
        externalSession.pendingClarification = {
          anchorUser: anchorTurn?.user || null,
          at: new Date().toISOString()
        };
      } else {
        externalSession.pendingClarification = null;
      }
    }

    this.store.scheduleSave(['core', 'interactions', 'language', 'knowledge']);

    const trainerResult = this.runTrainerTick(15);

    return {
      reply,
      debug: {
        source,
        confidence,
        learnedFacts: Object.keys(this.store.state.learnedFacts).length,
        memories: this.store.state.interactions.length,
        concepts: Object.keys(this.store.state.conceptGraph).length,
        webSources: Object.keys(this.store.state.webKnowledge || {}).length,
        neuralPrototypes: this.ensureNeuralState().prototypes.length,
        memoryRef: null,
        trainerProcessed: this.store.state.stats.trainerProcessedInteractions,
        trainerRemaining: trainerResult.remaining
      }
    };
  }

  async chat({ sessionId, message, webContexts = [] }) {
    this.ensureSession(sessionId);

    const fact = this.maybeLearnFact(message);
    const contextText = webContexts.map((item) => item.text || '').join(' ');
    const contextAwareQuery = this.buildContextAwareQuery(sessionId, message);
    const concepts = this.extractConcepts(`${contextAwareQuery} ${contextText}`);
    let replyData;

    if (fact) {
      replyData = {
        text: `Got it. I will remember this: ${fact}`,
        source: 'fact_learning',
        confidence: 0.95
      };
    } else {
      const winner = this.chooseBestCandidate(contextAwareQuery, concepts, webContexts, sessionId);
      replyData = {
        text: winner.text,
        source: winner.source,
        confidence: winner.confidence,
        memoryRef: winner.memoryRef || null,
        neuralRef: winner.neuralRef || null,
        mathMeta: winner.mathMeta || null
      };

      const session = this.store.state.sessions[sessionId];
      const lastTurn =
        session && Array.isArray(session.recentTurns) && session.recentTurns.length
          ? session.recentTurns[session.recentTurns.length - 1]
          : null;

      if (lastTurn && lastTurn.bot) {
        const repeatedReplyScore = this.scoreSimilarity(replyData.text, lastTurn.bot);
        const userContinuityScore = this.scoreSimilarity(contextAwareQuery, lastTurn.user || '');
        const hasFollowupSignal = /(it|that|this|they|them|those|these|same|again|continue|more|why|how)/.test(
          this.normalize(message)
        );

        if (repeatedReplyScore >= 0.88 && userContinuityScore < 0.2 && !hasFollowupSignal) {
          replyData = {
            text: 'That seems like a new topic. Ask it directly (for example: "define apple"), and I will answer that instead of repeating the previous response.',
            source: 'clarification_needed',
            confidence: 0.82,
            memoryRef: null,
            neuralRef: null,
            mathMeta: null
          };
        }
      }
    }


    this.updateTokenGraph(this.tokenize(message));
    this.updateConceptGraph(concepts);
    this.updateAssociationGraph(concepts);
    this.rememberResponse(message, replyData.text);

    this.store.state.interactions.push({
      sessionId,
      user: message,
      bot: replyData.text,
      source: replyData.source,
      confidence: replyData.confidence,
      mathMeta: replyData.mathMeta || null,
      concepts,
      at: new Date().toISOString()
    });

    if (this.store.state.interactions.length > this.maxMemory) {
      const toTrim = this.store.state.interactions.length - this.maxMemory;
      this.store.state.interactions = this.store.state.interactions.slice(-this.maxMemory);
      this.store.state.trainer.processedUntil = Math.max(0, this.store.state.trainer.processedUntil - toTrim);
    }

    this.store.state.stats.messages += 1;
    this.store.state.sessions[sessionId].turns += 1;
    this.rememberSessionTurn(sessionId, message, replyData.text, replyData.source);

    const session = this.store.state.sessions[sessionId];
    if (session) {
      if (replyData.source === 'clarification_needed') {
        const anchorTurn = this.getLastContextAnchorTurn(sessionId);
        session.pendingClarification = {
          anchorUser: anchorTurn?.user || null,
          at: new Date().toISOString()
        };
      } else {
        session.pendingClarification = null;
      }
    }

    this.store.scheduleSave(['core', 'interactions', 'language', 'knowledge']);

    const trainerResult = this.runTrainerTick(15);

    debug('Reply source=%s confidence=%d session=%s', replyData.source, replyData.confidence, sessionId);

    return {
      reply: replyData.text,
      debug: {
        source: replyData.source,
        confidence: replyData.confidence,
        learnedFacts: Object.keys(this.store.state.learnedFacts).length,
        memories: this.store.state.interactions.length,
        concepts: Object.keys(this.store.state.conceptGraph).length,
        webSources: Object.keys(this.store.state.webKnowledge || {}).length,
        neuralPrototypes: this.ensureNeuralState().prototypes.length,
        memoryRef: replyData.memoryRef,
        neuralRef: replyData.neuralRef,
        trainerProcessed: this.store.state.stats.trainerProcessedInteractions,
        trainerRemaining: trainerResult.remaining
      }
    };
  }
}

module.exports = ChatbotBrain;