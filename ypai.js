class YPAI14 {
  /**
   * YPAI 1.4 — Модель глубокого обучения
   * @param {Object} config Конфигурация модели
   */
  constructor(config = {}) {
    // Параметры с значениями по умолчанию
    this.inputDim     = config.inputDim     || 128;
    this.hiddenDim    = config.hiddenDim    || 256;
    this.outputDim    = config.outputDim    || 10;
    this.numHeads     = config.numHeads     || 4;
    this.numLayers    = config.numLayers    || 3;
    this.maxSeqLen    = config.maxSeqLen    || 512;
    this.dropoutRate  = config.dropoutRate  || 0.1;

    // Проверка делимости размерности на количество голов
    if (this.hiddenDim % this.numHeads !== 0) {
      throw new Error(`hiddenDim должен быть кратен numHeads`);
    }
    this.dimPerHead = this.hiddenDim / this.numHeads;

    // Инициализация весов Xavier
    this.weights = {
      inputProj:  this.xavierMatrix(this.inputDim, this.hiddenDim),
      posEnc:     this.positionalEncoding(),
      layers:     Array.from({ length: this.numLayers }, () => ({
        attnQ: this.xavierMatrix(this.hiddenDim, this.hiddenDim),
        attnK: this.xavierMatrix(this.hiddenDim, this.hiddenDim),
        attnV: this.xavierMatrix(this.hiddenDim, this.hiddenDim),
        attnOut: this.xavierMatrix(this.hiddenDim, this.hiddenDim),
        ff1:    this.xavierMatrix(this.hiddenDim, this.hiddenDim * 4),
        ff2:    this.xavierMatrix(this.hiddenDim * 4, this.hiddenDim)
      })),
      outProj:    this.xavierMatrix(this.hiddenDim, this.outputDim)
    };

    // Смещения
    this.biases = {
      inputProj: new Array(this.hiddenDim).fill(0),
      layers: Array.from({ length: this.numLayers }, () => ({
        attnOut: new Array(this.hiddenDim).fill(0),
        ff1:     new Array(this.hiddenDim * 4).fill(0),
        ff2:     new Array(this.hiddenDim).fill(0)
      })),
      outProj: new Array(this.outputDim).fill(0)
    };

    // Сохранение для градиентов
    this.cache = {};
  }

  // === Инициализация весов Xavier ===
  xavierMatrix(rows, cols) {
    const limit = Math.sqrt(6 / (rows + cols));
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() - 0.5) * 2 * limit)
    );
  }

  // === Позиционное кодирование (синус/косинус) ===
  positionalEncoding() {
    const pe = Array.from({ length: this.maxSeqLen }, () => new Array(this.hiddenDim));
    for (let pos = 0; pos < this.maxSeqLen; pos++) {
      for (let i = 0; i < this.hiddenDim; i += 2) {
        const div = Math.pow(10000, -(i / this.hiddenDim));
        pe[pos][i]   = Math.sin(pos * div);
        pe[pos][i+1] = Math.cos(pos * div);
      }
    }
    return pe;
  }

  // === Матричные операции ===
  matMul(A, B) {
    const rA = A.length, cA = A[0].length, cB = B[0].length;
    const res = Array.from({ length: rA }, () => new Array(cB).fill(0));
    for (let i = 0; i < rA; i++) {
      for (let k = 0; k < cA; k++) {
        if (A[i][k] === 0) continue;
        for (let j = 0; j < cB; j++) {
          res[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return res;
  }

  addBias(mat, bias) {
    return mat.map(row => row.map((v, i) => v + bias[i]));
  }

  dropout(x, train = true) {
    if (!train || this.dropoutRate === 0) return [...x];
    const mask = x.map(row => row.map(() => Math.random() > this.dropoutRate));
    const scale = 1 / (1 - this.dropoutRate);
    return x.map((row, i) => row.map((v, j) => mask[i][j] ? v * scale : 0));
  }

  // === Активации ===
  relu(x)  { return Math.max(0, x); }
  reluDeriv(x) { return x > 0 ? 1 : 0; }
  sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, x)))); }
  
  softmax(row) {
    const max = Math.max(...row);
    const exps = row.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  layerNorm(x) {
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    const varr = x.reduce((a, b) => a + (b - mean)**2, 0) / x.length;
    const eps = 1e-6;
    const invStd = 1 / Math.sqrt(varr + eps);
    return {
      out: x.map(v => (v - mean) * invStd),
      mean, invStd
    };
  }

  // === Многоголовое внимание с маской ===
  multiHeadAttention(x, layerIdx, train = true) {
    const seqLen = x.length;
    const l = this.weights.layers[layerIdx];

    const Q = this.matMul(x, l.attnQ);
    const K = this.matMul(x, l.attnK);
    const V = this.matMul(x, l.attnV);

    // Разбиение по головам
    const split = (m) => Array.from({ length: this.numHeads }, (_, h) =>
      m.map(row => row.slice(h * this.dimPerHead, (h + 1) * this.dimPerHead))
    );

    const Qh = split(Q), Kh = split(K), Vh = split(V);
    const headsOut = [];

    for (let h = 0; h < this.numHeads; h++) {
      const q = Qh[h], k = Kh[h], v = Vh[h];
      const scale = 1 / Math.sqrt(this.dimPerHead);

      // Сходство
      const scores = q.map((qi) =>
        k.map((kj) => qi.reduce((s, v, i) => s + v * kj[i], 0) * scale)
      );

      // Маска будущих токенов (каузальная)
      for (let i = 0; i < seqLen; i++) {
        for (let j = i + 1; j < seqLen; j++) {
          scores[i][j] = -1e9;
        }
      }

      const attnWeights = scores.map(r => this.softmax(r));
      const headOut = Array.from({ length: seqLen }, () => new Array(this.dimPerHead).fill(0));
      
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          const w = attnWeights[i][j];
          for (let d = 0; d < this.dimPerHead; d++) {
            headOut[i][d] += w * v[j][d];
          }
        }
      }
      headsOut.push(headOut);
    }

    // Объединение голов
    let concat = Array.from({ length: seqLen }, (_, i) =>
      headsOut.flatMap(h => h[i])
    );

    concat = this.matMul(concat, l.attnOut);
    concat = this.addBias(concat, this.biases.layers[layerIdx].attnOut);
    return this.dropout(concat, train);
  }

  // === Трансформер-блок ===
  blockForward(x, layerIdx, train = true) {
    // Внимание + остаток + норма
    const attn = this.multiHeadAttention(x, layerIdx, train);
    const res1 = x.map((row, i) => row.map((v, j) => v + attn[i][j]));
    const norm1 = res1.map(row => this.layerNorm(row));
    const n1 = norm1.map(n => n.out);

    // Прямая сеть
    const l = this.weights.layers[layerIdx];
    let ff = this.matMul(n1, l.ff1);
    ff = this.addBias(ff, this.biases.layers[layerIdx].ff1);
    ff = ff.map(row => row.map(this.relu));
    ff = this.dropout(ff, train);
    ff = this.matMul(ff, l.ff2);
    ff = this.addBias(ff, this.biases.layers[layerIdx].ff2);
    ff = this.dropout(ff, train);

    const res2 = n1.map((row, i) => row.map((v, j) => v + ff[i][j]));
    return res2.map(row => this.layerNorm(row).out);
  }

  // === Прямой проход ===
  forward(inputSeq, train = false) {
    const seqLen = inputSeq.length;
    if (seqLen > this.maxSeqLen) throw new Error('Слишком длинная последовательность');

    // Входная проекция + позиционное кодирование
    let x = this.matMul(inputSeq, this.weights.inputProj);
    x = this.addBias(x, this.biases.inputProj);
    
    for (let pos = 0; pos < seqLen; pos++) {
      for (let d = 0; d < this.hiddenDim; d++) {
        x[pos][d] += this.weights.posEnc[pos][d];
      }
    }

    // Блоки
    for (let i = 0; i < this.numLayers; i++) {
      x = this.blockForward(x, i, train);
    }

    // Финальный пуллинг (по последнему токену)
    const last = [x.at(-1)];
    let out = this.matMul(last, this.weights.outProj);
    out = this.addBias(out, this.biases.outProj);
    
    return out[0].map(this.sigmoid);
  }

  // === Шаг обучения (псевдо-обратное распространение, упрощённое) ===
  trainStep(inputSeq, target, lr = 0.005) {
    const pred = this.forward(inputSeq, true);
    const loss = pred.reduce((s, p, i) => s + (p - target[i]) ** 2, 0) / target.length;

    // Простейшее обновление весов (для примера)
    const grad = pred.map((p, i) => 2 * (p - target[i]) / target.length);
    
    // Обновление выходного слоя
    for (let i = 0; i < this.hiddenDim; i++) {
      for (let j = 0; j < this.outputDim; j++) {
        this.weights.outProj[i][j] -= lr * grad[j] * 0.1;
      }
    }
    for (let j = 0; j < this.outputDim; j++) {
      this.biases.outProj[j] -= lr * grad[j];
    }

    return loss;
  }
}

// ==========================================
// 🧪 ПРИМЕР ИСПОЛЬЗОВАНИЯ
// ==========================================
console.log('✅ Инициализация YPAI 1.4...');

const model = new YPAI14({
  inputDim:   16,    // Размер входа
  hiddenDim:  64,    // Скрытое пространство (кратно 4!)
  outputDim:  3,     // Количество классов/значений
  numHeads:   4,
  numLayers:  3,
  maxSeqLen:  100
});

// Вход: 8 временных шагов по 16 признаков
const input = Array.from({ length: 8 }, () =>
  Array.from({ length: 16 }, () => Math.random())
);

// Цель
const target = [1, 0, 0.5];

console.log('\n🔍 Первый прогон:');
const res = model.forward(input);
console.log('Выход:', res.map(v => v.toFixed(4)));

console.log('\n📉 Обучение (50 шагов):');
for (let epoch = 1; epoch <= 50; epoch++) {
  const loss = model.trainStep(input, target, 0.02);
  if (epoch % 10 === 0) {
    console.log(`Эпоха ${epoch} | Потеря: ${loss.toFixed(6)}`);
  }
}

console.log('\n🎯 Результат после обучения:');
const final = model.forward(input);
console.log('Выход:', final.map(v => v.toFixed(4)));
console.log('Цель: ', target);
