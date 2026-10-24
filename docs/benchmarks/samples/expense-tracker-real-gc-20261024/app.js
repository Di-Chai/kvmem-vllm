"use strict";

/* ============================================================
 * 本地记账本
 * 功能：新增/删除收支、月份筛选、本月收支与余额统计
 * 数据：localStorage 持久化，刷新保留，无外网依赖
 * ============================================================ */

const STORAGE_KEY = "expense-tracker:records";

/** 当前筛选的月份，null 表示显示全部 */
let currentMonthFilter = null;

/** 当前备注关键词，空字符串表示不筛选 */
let currentKeyword = "";

/* ---------- 工具函数 ---------- */

/** 生成唯一 id */
function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

/** 今天日期，格式 YYYY-MM-DD */
function todayStr() {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return d.getFullYear() + "-" + m + "-" + day;
}

/** 当前月份，格式 YYYY-MM */
function currentMonthStr() {
  return todayStr().slice(0, 7);
}

/** 金额格式化：¥1,234.56 */
function formatMoney(n) {
  const num = Number(n) || 0;
  const fixed = Math.abs(num).toFixed(2);
  const [intPart, decPart] = fixed.split(".");
  const withComma = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return (num < 0 ? "-¥" : "¥") + withComma + "." + decPart;
}

/* ---------- 数据读写（localStorage） ---------- */

/** 读取全部记录，返回数组；数据损坏时返回空数组 */
function loadRecords() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const data = JSON.parse(raw);
    if (!Array.isArray(data)) return [];
    return data.filter(
      (r) =>
        r &&
        typeof r.id === "string" &&
        (r.type === "income" || r.type === "expense") &&
        typeof r.amount === "number" &&
        r.amount > 0 &&
        typeof r.date === "string"
    );
  } catch (e) {
    console.warn("读取本地数据失败，已重置：", e);
    return [];
  }
}

/** 保存全部记录 */
function saveRecords(records) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch (e) {
    console.warn("保存本地数据失败：", e);
    alert("保存失败：浏览器 localStorage 不可用或已满，数据可能未持久化。");
  }
}

/* ---------- 渲染 ---------- */

/** 渲染统计卡片（始终基于“本月”） */
function renderStats() {
  const records = loadRecords();
  const month = currentMonthStr();

  let income = 0;
  let expense = 0;
  for (const r of records) {
    if (r.date.slice(0, 7) !== month) continue;
    if (r.type === "income") income += r.amount;
    else expense += r.amount;
  }

  const balance = income - expense;

  const incomeEl = document.getElementById("total-income");
  const expenseEl = document.getElementById("total-expense");
  const balanceEl = document.getElementById("balance");

  incomeEl.textContent = formatMoney(income);
  expenseEl.textContent = formatMoney(expense);
  balanceEl.textContent = formatMoney(balance);
  balanceEl.classList.toggle("negative", balance < 0);
}

/** 渲染记录列表（按月份 + 备注关键词筛选，日期倒序） */
function renderList() {
  const listEl = document.getElementById("record-list");
  const emptyEl = document.getElementById("empty-tip");
  const infoEl = document.getElementById("filter-info");

  let records = loadRecords();
  const total = records.length;

  // 月份筛选
  if (currentMonthFilter) {
    records = records.filter((r) => r.date.slice(0, 7) === currentMonthFilter);
  }

  // 备注关键词筛选（不区分大小写）
  if (currentKeyword) {
    const kw = currentKeyword.toLowerCase();
    records = records.filter((r) => (r.note || "").toLowerCase().includes(kw));
  }

  // 筛选信息提示
  if (currentMonthFilter && currentKeyword) {
    infoEl.textContent =
      "已筛选 " + currentMonthFilter + " + 关键词「" + currentKeyword + "」，共 " + records.length + " 条";
  } else if (currentMonthFilter) {
    infoEl.textContent = "已筛选 " + currentMonthFilter + "，共 " + records.length + " 条";
  } else if (currentKeyword) {
    infoEl.textContent = "关键词「" + currentKeyword + "」，共 " + records.length + " 条（全部月份）";
  } else {
    infoEl.textContent = "共 " + total + " 条记录";
  }

  // 日期倒序，同日按创建顺序（id 含时间戳）倒序
  records.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : a.id < b.id ? 1 : -1));

  listEl.innerHTML = "";
  for (const r of records) {
    const li = document.createElement("li");
    li.className = "record-item";
    li.dataset.id = r.id;

    const typeTag = document.createElement("span");
    typeTag.className = "record-type " + r.type;
    typeTag.textContent = r.type === "income" ? "收入" : "支出";

    const main = document.createElement("div");
    main.className = "record-main";

    const note = document.createElement("div");
    note.className = "record-note";
    note.textContent = r.note ? r.note : (r.type === "income" ? "收入" : "支出");

    const date = document.createElement("div");
    date.className = "record-date";
    date.textContent = r.date;

    main.appendChild(note);
    main.appendChild(date);

    const amount = document.createElement("div");
    amount.className = "record-amount " + r.type;
    amount.textContent = (r.type === "income" ? "+" : "-") + formatMoney(r.amount);

    const delBtn = document.createElement("button");
    delBtn.className = "btn-delete";
    delBtn.textContent = "删除";
    delBtn.title = "删除这条记录";
    delBtn.addEventListener("click", () => deleteRecord(r.id));

    li.appendChild(typeTag);
    li.appendChild(main);
    li.appendChild(amount);
    li.appendChild(delBtn);
    listEl.appendChild(li);
  }

  emptyEl.hidden = records.length > 0;
}

/** 刷新全部视图 */
function renderAll() {
  renderStats();
  renderList();
}

/* ---------- 操作 ---------- */

/** 新增一条记录 */
function addRecord(data) {
  const records = loadRecords();
  records.push({
    id: genId(),
    type: data.type,
    amount: data.amount,
    date: data.date,
    note: data.note
  });
  saveRecords(records);
  renderAll();
}

/** 删除一条记录 */
function deleteRecord(id) {
  const records = loadRecords();
  const target = records.find((r) => r.id === id);
  if (!target) return;
  if (!confirm("确定删除这条记录吗？\n" + target.date + " " + (target.note || "") + " " + formatMoney(target.amount))) {
    return;
  }
  saveRecords(records.filter((r) => r.id !== id));
  renderAll();
}

/** 设置月份筛选（null 为全部） */
function setMonthFilter(month) {
  currentMonthFilter = month;
  renderList();
}

/** 设置备注关键词（空字符串为全部） */
function setKeyword(keyword) {
  currentKeyword = keyword;
  renderList();
}

/** 清空搜索与月份筛选，恢复全部 */
function clearFilters() {
  const monthFilter = document.getElementById("month-filter");
  const keywordSearch = document.getElementById("keyword-search");
  monthFilter.value = "";
  keywordSearch.value = "";
  currentMonthFilter = null;
  currentKeyword = "";
  renderList();
}

/* ---------- 事件绑定与初始化 ---------- */

function init() {
  // 日期默认今天
  const dateInput = document.getElementById("date");
  dateInput.value = todayStr();

  // 月份筛选控件
  const monthFilter = document.getElementById("month-filter");
  monthFilter.addEventListener("change", () => {
    setMonthFilter(monthFilter.value ? monthFilter.value : null);
  });

  // 备注关键词搜索
  const keywordSearch = document.getElementById("keyword-search");
  keywordSearch.addEventListener("input", () => {
    setKeyword(keywordSearch.value.trim());
  });

  // 清空搜索（同时清空关键词与月份筛选）
  document.getElementById("search-clear").addEventListener("click", () => {
    clearFilters();
  });

  // 显示全部（仅重置月份，保留关键词）
  document.getElementById("filter-all").addEventListener("click", () => {
    monthFilter.value = "";
    setMonthFilter(null);
  });

  // 新增表单
  const form = document.getElementById("record-form");
  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const type = document.getElementById("type").value;
    const amountRaw = document.getElementById("amount").value;
    const date = dateInput.value;
    const note = document.getElementById("note").value.trim();

    const amount = Math.round(parseFloat(amountRaw) * 100) / 100;
    if (!amount || amount <= 0) {
      alert("请输入大于 0 的金额");
      return;
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      alert("请选择有效日期");
      return;
    }

    addRecord({ type, amount, date, note });

    // 清空金额与备注，保留类型和日期，便于连续录入
    document.getElementById("amount").value = "";
    document.getElementById("note").value = "";
    document.getElementById("amount").focus();
  });

  renderAll();
}

document.addEventListener("DOMContentLoaded", init);
