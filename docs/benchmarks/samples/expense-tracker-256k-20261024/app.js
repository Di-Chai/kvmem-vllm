"use strict";

/* ============================================================
 * 个人记账本 - 逻辑
 * 数据保存在 localStorage，刷新后保留。
 * ============================================================ */

const STORAGE_KEY = "expense-tracker-records-v1";

/** @typedef {{id:string, type:"income"|"expense", amount:number, desc:string, date:string}} Record */
/** @type {Record[]} */
let records = [];

/** 当前筛选的月份，格式 "YYYY-MM"；null 表示"全部" */
let currentMonth = null;

// ---------- 工具函数 ----------

function todayStr() {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

function currentMonthStr() {
  return todayStr().slice(0, 7); // "YYYY-MM"
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function formatMoney(n) {
  const sign = n < 0 ? "-" : "";
  return sign + "¥" + Math.abs(n).toLocaleString("zh-CN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function monthLabel(ym) {
  const [y, m] = ym.split("-");
  return `${y} 年 ${parseInt(m, 10)} 月`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------- 数据持久化 ----------

function loadRecords() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // 过滤掉不合法的数据
    return parsed.filter(
      (r) =>
        r &&
        typeof r.id === "string" &&
        (r.type === "income" || r.type === "expense") &&
        typeof r.amount === "number" &&
        isFinite(r.amount) &&
        r.amount > 0 &&
        typeof r.desc === "string" &&
        typeof r.date === "string" &&
        /^\d{4}-\d{2}-\d{2}$/.test(r.date)
    );
  } catch (e) {
    console.error("读取 localStorage 失败:", e);
    return [];
  }
}

function saveRecords() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch (e) {
    console.error("写入 localStorage 失败:", e);
    alert("保存失败：localStorage 不可用或被占用。");
  }
}

// ---------- 统计 ----------

/** 计算某月份（"YYYY-MM"）的收入、支出、余额 */
function calcMonthStats(ym) {
  let income = 0;
  let expense = 0;
  for (const r of records) {
    if (r.date.slice(0, 7) !== ym) continue;
    if (r.type === "income") income += r.amount;
    else expense += r.amount;
  }
  return { income, expense, balance: income - expense };
}

// ---------- 渲染 ----------

function renderStats() {
  const stats = calcMonthStats(currentMonthStr());
  document.getElementById("stat-income").textContent = formatMoney(stats.income);
  document.getElementById("stat-expense").textContent = formatMoney(stats.expense);
  const balEl = document.getElementById("stat-balance");
  balEl.textContent = formatMoney(stats.balance);
  balEl.style.color = stats.balance < 0 ? "var(--expense)" : "var(--accent)";
}

function renderMonthFilter() {
  const sel = document.getElementById("month-filter");
  const months = new Set(records.map((r) => r.date.slice(0, 7)));
  months.add(currentMonthStr()); // 至少包含当前月
  const sorted = Array.from(months).sort().reverse();

  const prevValue = sel.value || null;
  sel.innerHTML = "";

  const allOpt = document.createElement("option");
  allOpt.value = "";
  allOpt.textContent = "全部月份";
  sel.appendChild(allOpt);

  for (const ym of sorted) {
    const opt = document.createElement("option");
    opt.value = ym;
    opt.textContent = monthLabel(ym);
    sel.appendChild(opt);
  }

  // 恢复之前的选择（若仍有效）
  if (prevValue && sorted.includes(prevValue)) {
    sel.value = prevValue;
    currentMonth = prevValue;
  } else if (currentMonth && sorted.includes(currentMonth)) {
    sel.value = currentMonth;
  } else {
    sel.value = "";
    currentMonth = null;
  }
}

function renderList() {
  const list = document.getElementById("record-list");
  const title = document.getElementById("list-title");

  let filtered = records;
  if (currentMonth) {
    filtered = records.filter((r) => r.date.slice(0, 7) === currentMonth);
    title.textContent = `记录列表（${monthLabel(currentMonth)}）`;
  } else {
    title.textContent = "记录列表（全部）";
  }

  if (filtered.length === 0) {
    list.innerHTML =
      '<div class="empty">暂无记录，先在上方添加一条吧。</div>';
    return;
  }

  // 按日期倒序，同日按创建顺序倒序
  const sorted = filtered
    .map((r, i) => ({ r, i }))
    .sort((a, b) => (b.r.date === a.r.date ? b.i - a.i : b.r.date < a.r.date ? -1 : 1));

  list.innerHTML = sorted
    .map(({ r }) => {
      const isIncome = r.type === "income";
      const sign = isIncome ? "+" : "-";
      return `
        <div class="record">
          <span class="badge ${r.type}">${isIncome ? "收入" : "支出"}</span>
          <div class="info">
            <div class="desc">${escapeHtml(r.desc)}</div>
            <div class="date">${r.date}</div>
          </div>
          <span class="amt ${r.type}">${sign}${formatMoney(r.amount)}</span>
          <button class="btn-del" data-id="${r.id}" title="删除">删除</button>
        </div>`;
    })
    .join("");

  list.querySelectorAll(".btn-del").forEach((btn) => {
    btn.addEventListener("click", () => deleteRecord(btn.dataset.id));
  });
}

function renderAll() {
  renderStats();
  renderMonthFilter();
  renderList();
}

// ---------- 操作 ----------

function addRecord(type, amount, desc, date) {
  records.push({ id: uid(), type, amount, desc, date });
  saveRecords();
  renderAll();
}

function deleteRecord(id) {
  const rec = records.find((r) => r.id === id);
  if (!rec) return;
  if (!confirm(`确定删除这条记录吗？\n${rec.date} ${rec.type === "income" ? "收入" : "支出"} ${formatMoney(rec.amount)} ${rec.desc}`)) {
    return;
  }
  records = records.filter((r) => r.id !== id);
  saveRecords();
  renderAll();
}

// ---------- 事件绑定 ----------

function init() {
  records = loadRecords();

  // 日期默认今天
  document.getElementById("date").value = todayStr();

  // 新增表单
  document.getElementById("add-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const errEl = document.getElementById("form-error");
    errEl.textContent = "";

    const type = document.getElementById("type").value;
    const amountRaw = document.getElementById("amount").value;
    const desc = document.getElementById("desc").value.trim();
    const date = document.getElementById("date").value;

    const amount = Number(amountRaw);
    if (!amountRaw || !isFinite(amount) || amount <= 0) {
      errEl.textContent = "请输入有效的金额（大于 0）。";
      return;
    }
    if (amount > 1e12) {
      errEl.textContent = "金额过大，请检查。";
      return;
    }
    if (!desc) {
      errEl.textContent = "请填写描述。";
      return;
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      errEl.textContent = "请选择有效的日期。";
      return;
    }

    addRecord(type, Math.round(amount * 100) / 100, desc, date);

    // 清空金额和描述，保留类型和日期，方便连续录入
    document.getElementById("amount").value = "";
    document.getElementById("desc").value = "";
    document.getElementById("amount").focus();
  });

  // 月份筛选
  document.getElementById("month-filter").addEventListener("change", (e) => {
    currentMonth = e.target.value || null;
    renderList();
  });

  renderAll();
}

document.addEventListener("DOMContentLoaded", init);
