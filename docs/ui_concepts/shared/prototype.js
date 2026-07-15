const option = document.body.dataset.option || 'c';
const pageKey = new URLSearchParams(window.location.search).get('page') || 'home';

const optionNames = {
  a: '方案 A · 临床工作站',
  b: '方案 B · 研究分析台',
  c: '方案 C · 现代模型中转台',
  hybrid: '推荐混合方案',
};

const pageMeta = {
  home: ['中转台总览', '模型资产、任务接入和研究评测的统一入口'],
  assets: ['模型资产目录', '按 checkpoint 审视来源、模态、访问方式与接入准备度'],
  readiness: ['任务模型与接入状态', '状态逐级独立展示；基础加载通过不代表任务推理或路由可用'],
  research: ['研究评测', '冻结协议、评测入口和最近运行集中管理'],
  import: ['结果表上传与字段映射', '导入、识别、确认、校验四步完成离线评测准备'],
  overview: ['风险审计 · 总体表现', '59 类虚构演示数据 · 主预测版本 base'],
  risk: ['风险审计 · 错误风险', '分析模型输出错误与固定复核预算，不推断临床后果'],
  cases: ['风险审计 · 病例清单', '仅显示会话序号与模型输出风险信号'],
  leakage: ['数据与泄漏检查', '只报告结果表可观察证据，不声称已排除数据泄漏'],
};

const logo = `
<svg viewBox="0 0 128 128" role="img" aria-label="OphAgent Retina Router">
  <defs>
    <linearGradient id="mark-bg" x1="18" y1="14" x2="112" y2="118" gradientUnits="userSpaceOnUse">
      <stop stop-color="#1D4ED8"/><stop offset=".52" stop-color="#0F766E"/><stop offset="1" stop-color="#14B8A6"/>
    </linearGradient>
    <linearGradient id="mark-eye" x1="24" y1="58" x2="101" y2="72" gradientUnits="userSpaceOnUse">
      <stop stop-color="#E0F2FE"/><stop offset=".5" stop-color="#FFFFFF"/><stop offset="1" stop-color="#CCFBF1"/>
    </linearGradient>
  </defs>
  <rect x="10" y="10" width="108" height="108" rx="28" fill="url(#mark-bg)"/>
  <path d="M25 65C34 50 47 42 64 42C81 42 94 50 103 65C94 80 81 88 64 88C47 88 34 80 25 65Z" fill="url(#mark-eye)"/>
  <path d="M34 65C42 55 52 50 64 50C76 50 86 55 94 65C86 75 76 80 64 80C52 80 42 75 34 65Z" fill="#0F172A" fill-opacity=".14"/>
  <circle cx="64" cy="65" r="15" fill="#0F172A" fill-opacity=".76"/><circle cx="64" cy="65" r="8" fill="#5EEAD4"/><circle cx="60" cy="61" r="3.4" fill="#FFFFFF" fill-opacity=".88"/>
  <path d="M78 52H96C101 52 104 55 104 60V62M78 78H96C101 78 104 75 104 70V68M80 65H108" stroke="#FFFFFF" stroke-width="5" stroke-linecap="round"/>
  <circle cx="104" cy="60" r="6" fill="#F59E0B" stroke="#FFFFFF" stroke-width="3"/><circle cx="108" cy="65" r="6" fill="#60A5FA" stroke="#FFFFFF" stroke-width="3"/><circle cx="104" cy="70" r="6" fill="#34D399" stroke="#FFFFFF" stroke-width="3"/>
</svg>`;

const navGroups = [
  ['模型中转台', [
    ['home', '总览'], ['assets', '模型资产'], ['readiness', '任务模型与接入'],
  ]],
  ['研究评测', [
    ['research', '评测主页'], ['import', '结果表导入'], ['overview', '总体表现'], ['risk', '错误风险'], ['cases', '病例清单'], ['leakage', '数据检查'],
  ]],
  ['回放与记录', [
    ['replay', '病例回放'], ['runs', '任务运行记录'],
  ]],
];

function navHtml() {
  let index = 1;
  return navGroups.map(([label, links]) => `
    <div class="nav-group">
      <div class="nav-label">${label}</div>
      ${links.map(([key, text]) => {
        const number = String(index++).padStart(2, '0');
        const href = pageMeta[key] ? `?page=${key}` : '?page=home';
        return `<a class="nav-link ${key === pageKey ? 'active' : ''}" href="${href}"><span class="nav-index">${number}</span><span>${text}</span></a>`;
      }).join('')}
    </div>`).join('');
}

function shell(content) {
  const [title] = pageMeta[pageKey] || pageMeta.home;
  return `
    <div class="app-shell">
      <aside class="sidebar">
        <a class="brand" href="?page=home">
          <span class="brand-mark">${logo}</span>
          <span class="brand-copy"><strong>OphAgent</strong><span>Model Hub · 研究环境</span></span>
        </a>
        <nav class="nav" aria-label="主导航">${navHtml()}</nav>
        <div class="sidebar-foot"><strong>系统边界</strong>模型输出研究审计，不提供诊断或患者分流决定。</div>
      </aside>
      <div class="workspace">
        <header class="topbar">
          <div class="crumb">眼科模型中转台 / <strong>${title}</strong></div>
          <div class="top-actions"><span class="env-badge">演示数据</span><span class="env-badge">服务正常</span></div>
        </header>
        <main class="page">${content}<div class="prototype-note">隔离设计原型 · ${optionNames[option]} · 不连接生产数据与计算逻辑</div></main>
      </div>
    </div>`;
}

function pageHead(eyebrow, actions = '') {
  const [title, subtitle] = pageMeta[pageKey] || pageMeta.home;
  return `<div class="page-head"><div><p class="eyebrow">${eyebrow}</p><h1>${title}</h1><p class="page-subtitle">${subtitle}</p></div><div class="page-head-actions">${actions}</div></div>`;
}

function status(text, kind = 'unknown') { return `<span class="status ${kind}">${text}</span>`; }
function button(text, kind = '', extra = '') { return `<button class="btn ${kind}" type="button" ${extra}>${text}</button>`; }
function field(label, value, cls = '') { return `<div class="field ${cls}"><label>${label}</label><select class="input"><option>${value}</option></select></div>`; }

function kpis(items) {
  return `<div class="kpi-grid">${items.map(([label, value, meta, kind = '']) => `
    <div class="kpi ${kind}"><span class="kpi-label">${label}</span><strong class="kpi-value">${value}</strong><span class="kpi-meta">${meta}</span></div>`).join('')}</div>`;
}

function auditTabs(active) {
  const tabs = [['overview', '总体表现'], ['risk', '错误风险'], ['cases', '病例清单'], ['stability', '版本稳定性'], ['leakage', '数据与泄漏检查']];
  return `<nav class="audit-tabs" aria-label="审计结果视图">${tabs.map(([key, text]) => `<a class="audit-tab ${active === key ? 'active' : ''}" href="${key === 'stability' ? '?page=overview' : `?page=${key}`}">${text}</a>`).join('')}</nav>`;
}

function homePage() {
  return `${pageHead('MODEL HUB / CONTROL', button('刷新状态', '') + button('导入结果表', 'primary', 'onclick="location.href=\'?page=import\'"'))}
    <div class="boundary compact"><strong>研究边界</strong><span>模型工程与评测用于研发验证；病例回放只展示模型调用轨迹。</span></div>
    ${kpis([
      ['登记模型', '15', 'OphBench Registry'], ['当前 CFP 候选', '13', '已排除范围外资产'], ['任务推理可用', '1', '标准类别概率'], ['可进入路由池', '1', '评测与成本门控'], ['离线审计产物', '4', '最近 30 天'], ['待处理任务', '2', '1 个需要确认', 'attention'],
    ])}
    <section class="section"><div class="section-head"><div><h2>当前任务流</h2><p>从资产登记到路由资格，状态不跨级推导。</p></div><span>${status('DR 五级分级', 'info')}</span></div>
      <div class="workflow">
        <div class="workflow-step"><small>01 模型资产</small><strong>13 个 CFP 候选</strong><span class="cell-sub">来源与访问限制已登记</span></div>
        <div class="workflow-step"><small>02 接入准备</small><strong>1 个基础加载通过</strong><span class="cell-sub">12 个 Adapter 待实现</span></div>
        <div class="workflow-step current"><small>03 研究评测</small><strong>2 个运行可查看</strong><span class="cell-sub">路由组合 + 结果表审计</span></div>
        <div class="workflow-step"><small>04 路由资格</small><strong>1 个可进入路由池</strong><span class="cell-sub">统一评测与成本已核验</span></div>
      </div>
    </section>
    <div class="split-2">
      <section class="section"><div class="section-head"><div><h2>最近工作</h2><p>面向继续操作，而不是重复展示总数。</p></div><a class="action-link" href="?page=research">查看全部</a></div>
        ${simpleRunTable()}
      </section>
      <section class="section"><div class="section-head"><div><h2>需要关注</h2><p>只呈现阻塞项与边界提示。</p></div></div>
        <div class="check-list">
          <div class="check-row" style="grid-template-columns:1fr 120px"><div><strong>12 个 checkpoint</strong><span class="cell-sub">Adapter 尚未实现</span></div>${status('暂不可路由', 'unknown')}</div>
          <div class="check-row" style="grid-template-columns:1fr 120px"><div><strong>1 个上传产物</strong><span class="cell-sub">可离线审计，不代表在线接入</span></div>${status('边界明确', 'good')}</div>
          <div class="check-row" style="grid-template-columns:1fr 120px"><div><strong>临床后果风险</strong><span class="cell-sub">暂无医生规则或严重度证据</span></div>${status('尚未评估', 'unknown')}</div>
        </div>
      </section>
    </div>`;
}

function simpleRunTable() {
  return `<div class="table-wrap"><table><thead><tr><th style="width:32%">运行</th><th>类型</th><th>状态</th><th>更新时间</th></tr></thead><tbody>
    <tr><td data-label="运行"><span class="cell-title">FRD59 输出审计</span><span class="cell-sub">主版本 base</span></td><td data-label="类型">结果表审计</td><td data-label="状态">${status('已完成', 'good')}</td><td data-label="更新时间">14:32</td></tr>
    <tr><td data-label="运行"><span class="cell-title">DR 路由扫描</span><span class="cell-sub">预算 0–50%</span></td><td data-label="类型">组合评测</td><td data-label="状态">${status('已冻结', 'info')}</td><td data-label="更新时间">昨天</td></tr>
    <tr><td data-label="运行"><span class="cell-title">RETFound 接入核验</span><span class="cell-sub">CFP checkpoint</span></td><td data-label="类型">接入准备</td><td data-label="状态">${status('待任务适配', 'warn')}</td><td data-label="更新时间">07-12</td></tr>
  </tbody></table></div>`;
}

const assets = [
  ['RETFound CFP', 'ViT-L/16', 'CFP', '官方 checkpoint', '需认证', '基础加载通过', 'task-ready'],
  ['DINOv2 Large', 'ViT-L/14', 'CFP', '基础编码器', '公开', 'Adapter 待实现', 'adapter'],
  ['KeepFIT', '双编码器', 'CFP / FFA / text', '基础编码器', '需认证', '权重待获取', 'asset'],
  ['MIRAGE', '多模态 Transformer', 'OCT / SLO', '基础编码器', '需申请', '模态不匹配', 'unknown'],
  ['FMUE OCT-16', 'ViT-B', 'OCT', '任务 checkpoint', '公开', '非当前任务', 'unknown'],
];

function assetTable() {
  return `<div class="table-wrap"><table><thead><tr><th style="width:19%">模型 / checkpoint</th><th style="width:13%">架构</th><th style="width:15%">输入模态</th><th style="width:15%">资产类型</th><th style="width:12%">访问</th><th style="width:16%">接入状态</th><th>操作</th></tr></thead><tbody>
    ${assets.map(([name, arch, modality, type, access, readiness, key]) => `<tr>
      <td data-label="模型"><span class="cell-title">${name}</span><span class="cell-sub">checkpoint 级记录</span></td><td data-label="架构">${arch}</td><td data-label="模态">${modality}</td><td data-label="类型">${type}</td><td data-label="访问">${access}</td><td data-label="状态">${status(readiness, key === 'task-ready' ? 'good' : key === 'adapter' ? 'warn' : 'unknown')}</td><td data-label="操作"><a class="action-link" href="?page=readiness">查看状态</a></td>
    </tr>`).join('')}
  </tbody></table></div>`;
}

function assetsPage() {
  return `${pageHead('MODEL CATALOG', button('导出当前视图') + button('刷新目录', 'primary'))}
    <div class="boundary compact"><strong>候选范围</strong><span>当前为 CFP 单模态任务；VisionFM legacy、生成模型及模态不匹配 checkpoint 不计入候选数。</span></div>
    ${kpis([['模型记录', '15', '上游 Registry'], ['可交接 checkpoint', '19', '范围筛选后'], ['当前 CFP 候选', '13', '基础模型与任务模型'], ['来源已核验', '19', '不等于本地已下载'], ['Adapter 已实现', '1', 'checkpoint 级'], ['可路由', '1', '当前任务门控后']])}
    <section class="section">
      <div class="toolbar">${field('模态', 'CFP')}${field('资产类型', '全部类型')}${field('接入状态', '全部状态')}<div class="field"><label>模型搜索</label><input class="input" value="" placeholder="模型或 checkpoint"/></div>${button('清除', 'small')}</div>
      ${assetTable()}
    </section>`;
}

function readinessCell(text, kind) { return `<div>${status(text, kind)}</div>`; }
function readinessPage() {
  const rows = [
    ['RETFound CFP', 'retfound-cfp', ['已登记','来源已核验','加载通过','已适配','可路由'], ['good','good','good','good','good']],
    ['DINOv2 Large', 'dinov2-large', ['已登记','来源已核验','待实现','待完成','暂不可路由'], ['good','good','warn','unknown','unknown']],
    ['KeepFIT', 'keepfit-cfp', ['已登记','来源已核验','权重待获取','待完成','暂不可路由'], ['good','good','warn','unknown','unknown']],
    ['FMUE OCT-16', 'fmue-oct16', ['已登记','来源已核验','未测试','非当前任务','暂不可路由'], ['good','good','unknown','unknown','unknown']],
  ];
  return `${pageHead('TASK READINESS', button('切换任务') + button('查看准入规则', 'primary'))}
    <div class="boundary compact"><strong>当前任务</strong><span>DR 五级分级 · CFP · 标准类别概率。任一阶段通过都不会自动推出后一阶段。</span></div>
    <section class="section"><div class="section-head"><div><h2>Checkpoint 接入矩阵</h2><p>将技术准备度与中转台资格拆开阅读。</p></div><span>${status('1 / 13 可路由', 'info')}</span></div>
      <div class="readiness-grid">
        <div class="head">模型 checkpoint</div><div class="head">资产登记</div><div class="head">官方来源</div><div class="head">Adapter / 基础加载</div><div class="head">任务适配</div><div class="head">路由资格</div>
        ${rows.map(([name, id, cells, kinds]) => `<div class="readiness-model"><strong>${name}</strong><small>${id}</small></div>${cells.map((cell, i) => readinessCell(cell, kinds[i])).join('')}`).join('')}
      </div>
    </section>
    <div class="split-2"><section class="section"><div class="section-head"><div><h2>RETFound CFP · 证据链</h2><p>显示每个阶段的证据，不用单一“可用”覆盖。</p></div></div>
      <div class="check-list">
        <div class="check-row" style="grid-template-columns:180px 120px 1fr"><strong>基础加载</strong>${status('已通过', 'good')}<span class="check-detail">encoder smoke · embedding 输出维度已登记</span></div>
        <div class="check-row" style="grid-template-columns:180px 120px 1fr"><strong>任务推理</strong>${status('已验证', 'good')}<span class="check-detail">APTOS DR 五分类线性探针 · 标准概率产物</span></div>
        <div class="check-row" style="grid-template-columns:180px 120px 1fr"><strong>统一评测</strong>${status('已完成', 'good')}<span class="check-detail">冻结 split 与成本记录可追溯</span></div>
      </div></section>
      <section class="section"><div class="section-head"><div><h2>状态定义</h2><p>负责人查看时保留最小但必要的解释。</p></div></div>
        <div class="method-note"><strong>“基础加载通过”不是“任务推理可用”</strong><p>编码器能输出 embedding，仅说明基础 smoke 成功；仍需分类头或冻结预测、统一评测、成本和数据隔离证据。</p></div>
      </section></div>`;
}

function researchPage() {
  return `${pageHead('RESEARCH EVALUATION', button('查看运行记录') + button('新建评测', 'primary'))}
    <div class="boundary compact"><strong>评测口径</strong><span>当前所有结果均为研发验证；公开标签代理事件不进入在线路由。</span></div>
    <section class="section"><div class="section-head"><div><h2>选择评测工作区</h2><p>先按研究问题分流，再进入参数设置，减少首屏控件。</p></div></div>
      <div class="mode-grid">
        <div class="mode"><span class="mode-kicker">ROUTING COMPOSITION</span><h3>路由组合评测</h3><p>比较 scout、专家池、调用预算、成本和冻结标签依赖代理事件。</p>${button('进入组合评测', 'primary')}</div>
        <div class="mode"><span class="mode-kicker">RESULT ARTIFACT</span><h3>结果表风险审计</h3><p>导入 CSV / Excel，确认字段映射，分析错误、不确定性、版本稳定性与泄漏线索。</p>${button('导入结果表', 'primary', 'onclick="location.href=\'?page=import\'"')}</div>
      </div>
    </section>
    <div class="split-2"><section class="section"><div class="section-head"><div><h2>冻结研究上下文</h2><p>指标之前先说明数据、标签与主版本。</p></div></div>
      <div class="check-list">
        <div class="check-row" style="grid-template-columns:170px 1fr"><strong>任务协议</strong><span>DR 五级分级 · 主指标 QWK</span></div>
        <div class="check-row" style="grid-template-columns:170px 1fr"><strong>测试集隔离</strong><span>冻结 prediction，仅作最终评测</span></div>
        <div class="check-row" style="grid-template-columns:170px 1fr"><strong>主预测版本</strong><span>运行前选择，不按测试表现自动切换</span></div>
      </div></section>
      <section class="section"><div class="section-head"><div><h2>最近运行</h2><p>避免在首页重复完整任务表。</p></div></div>${simpleRunTable()}</section></div>`;
}

function importPage() {
  return `${pageHead('RESULT IMPORT', button('查看格式示例') + button('运行校验', 'primary'))}
    <div class="stepper"><div class="step done" data-step="1">上传文件</div><div class="step done" data-step="2">识别字段</div><div class="step current" data-step="3">确认映射</div><div class="step" data-step="4">校验并审计</div></div>
    <div class="split-2">
      <div>
        <section class="section"><div class="section-head"><div><h2>结果表</h2><p>虚构演示文件 · 不包含原始病例标识。</p></div>${status('68 行 · 19 列', 'info')}</div>
          <div class="upload-zone"><div><strong>拖放 CSV 或 Excel，或选择文件</strong><p>支持任意类别数、无概率结果和多个预测版本</p><span class="file-chip">ophagent_mock_predictions.csv · 6.2 KB</span></div></div>
        </section>
        <section class="section"><div class="section-head"><div><h2>字段映射</h2><p>系统给出候选；运行前由用户确认。</p></div>${status('识别到 2 个版本', 'good')}</div>
          <div class="mapping-grid">${field('病例标识', 'case_id')}${field('真实标签', 'true_label')}${field('数据划分', 'split')}${field('主预测版本', 'base')}</div>
          <div class="mapping-group"><h3>预测版本 1 · base</h3><div class="mapping-grid">${field('预测标签', 'pred_label_no_tta')}${field('置信度', 'confidence_no_tta')}</div><div class="prob-chips">${['prob_A','prob_B','prob_C','prob_D','prob_E','prob_F'].map(x => `<span class="chip">${x}</span>`).join('')}</div></div>
          <div class="mapping-group"><h3>预测版本 2 · tta</h3><div class="mapping-grid">${field('预测标签', 'pred_label_tta')}${field('置信度', 'confidence_tta')}</div></div>
        </section>
      </div>
      <aside>
        <section class="section"><div class="section-head"><div><h2>校验预览</h2><p>严重问题阻止审计，警告允许继续。</p></div></div>
          <div class="check-list">
            <div class="check-row" style="grid-template-columns:1fr 104px"><span>唯一病例</span><strong class="num">68</strong></div>
            <div class="check-row" style="grid-template-columns:1fr 104px"><span>类别数</span><strong class="num">6</strong></div>
            <div class="check-row" style="grid-template-columns:1fr 104px"><span>概率范围与行和</span>${status('通过', 'good')}</div>
            <div class="check-row" style="grid-template-columns:1fr 104px"><span>版本覆盖一致</span>${status('通过', 'good')}</div>
            <div class="check-row" style="grid-template-columns:1fr 104px"><span>患者级重叠</span>${status('无法评估', 'unknown')}</div>
          </div>
        </section>
        <div class="method-note"><strong>导入资格边界</strong><p>完成校验后只获得 offline_evaluation_eligible，不会获得 Adapter、在线推理或 route_eligible。</p></div>
      </aside>
    </div>`;
}

function auditKpis() {
  return kpis([
    ['样本数', '1,248', '测试 split'], ['类别数', '59', '自动识别'], ['Accuracy', '82.4%', '95% CI 80.2–84.5'], ['Macro-F1', '77.8%', '类别均衡观察'], ['错误病例', '220', '17.6%', 'attention'], ['高置信错误', '31', '阈值 ≥ 0.80', 'danger'],
  ]);
}

function auditHeader(active) {
  return `${pageHead('OUTPUT RISK AUDIT', button('下载结果') + button('审计设置', 'primary'))}
    <div class="boundary compact"><strong>分析边界</strong><span>这里分析模型输出错误风险；临床后果风险尚未评估。</span></div>${auditKpis()}${auditTabs(active)}`;
}

function overviewPage() {
  return `${auditHeader('overview')}
    <div class="split-2">
      <section class="section"><div class="section-head"><div><h2>主要混淆方向</h2><p>59 类默认显示 Top-10；可再聚焦单一真实类别。</p></div><span>${status('Top-10 / 59 类', 'info')}</span></div>
        <div class="chart"><h3 class="chart-title">错误病例的主要类别对</h3><p class="chart-sub">真实类别 → 预测类别 · 仅虚构演示数据</p>
          <div class="confusion-list">
            ${[['类别 07 → 类别 12',28,100],['类别 31 → 类别 14',22,79],['类别 04 → 类别 09',18,64],['类别 45 → 类别 46',15,54],['类别 18 → 类别 03',12,43],['类别 22 → 类别 21',9,32]].map(([label,n,w]) => `<div class="confusion-row"><label>${label}</label><div class="track"><span style="width:${w}%"></span></div><output>${n} 例</output></div>`).join('')}
          </div><div class="chart-callout">59 类时不渲染不可读的 59×59 全矩阵；保留 Top 混淆、类别搜索与下载完整矩阵。</div>
        </div>
      </section>
      <section class="section"><div class="section-head"><div><h2>总体指标</h2><p>先看总体，再定位长尾类别。</p></div></div>
        <div class="table-wrap"><table><thead><tr><th>指标</th><th class="num">值</th><th>解释</th></tr></thead><tbody>
          ${[['Accuracy','0.824','总体命中'],['Macro Precision','0.781','类别等权'],['Macro Recall','0.784','类别等权'],['Macro-F1','0.778','主要比较指标'],['Weighted F1','0.819','按样本数加权'],['Cohen Kappa','0.812','超越随机一致性']].map(row => `<tr><td data-label="指标">${row[0]}</td><td data-label="值" class="num">${row[1]}</td><td data-label="解释" class="cell-sub">${row[2]}</td></tr>`).join('')}
        </tbody></table></div>
      </section>
    </div>
    <section class="section"><div class="section-head"><div><h2>需要关注的类别</h2><p>按低 Recall 与 support 下限筛选，避免把 59 类一次铺满。</p></div>${button('查看全部 59 类', 'small')}</div>${classTable()}</section>`;
}

function classTable() {
  return `<div class="table-wrap"><table><thead><tr><th>类别</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th><th>观察</th></tr></thead><tbody>
    ${[['类别 07','.71','.58','.64','18','低 Recall'],['类别 31','.66','.62','.64','21','边界混淆'],['类别 45','.74','.67','.70','15','样本较少'],['类别 04','.79','.69','.74','27','需关注'],['类别 18','.83','.72','.77','22','需关注']].map(r => `<tr><td data-label="类别"><span class="cell-title">${r[0]}</span></td><td data-label="Precision" class="num">${r[1]}</td><td data-label="Recall" class="num">${r[2]}</td><td data-label="F1" class="num">${r[3]}</td><td data-label="Support" class="num">${r[4]}</td><td data-label="观察">${status(r[5], r[5] === '低 Recall' ? 'bad' : 'warn')}</td></tr>`).join('')}
  </tbody></table></div>`;
}

function riskPage() {
  return `${auditHeader('risk')}
    <div class="risk-summary">
      <div class="risk-item"><span>预测错误</span><strong>220</strong><span>占全部病例 17.6%</span></div>
      <div class="risk-item"><span>高置信错误</span><strong>31</strong><span>confidence ≥ 0.80</span></div>
      <div class="risk-item"><span>优先复核候选</span><strong>125</strong><span>10% 预算 · 按低 margin</span></div>
    </div>
    <div class="split-2" style="margin-top:16px">
      <section class="section"><div class="section-head"><div><h2>固定复核预算</h2><p>比较低 confidence、高 entropy 与低 margin。</p></div>${field('排序信号', '低 margin', 'narrow')}</div>
        <div class="chart"><h3 class="chart-title">预算增加时的错误捕获</h3><p class="chart-sub">虚构演示数据 · 阴影区不表示临床收益</p>
          <div class="line-chart"><svg viewBox="0 0 640 220" preserveAspectRatio="none" aria-label="固定复核预算错误捕获曲线"><polyline points="0,210 64,172 128,138 256,92 384,59 512,33 640,19" fill="none" stroke="var(--primary)" stroke-width="4"/><polyline points="0,210 64,196 128,181 256,151 384,122 512,92 640,64" fill="none" stroke="var(--amber)" stroke-width="3" stroke-dasharray="8 6"/></svg><span class="axis-label bottom">复核预算</span></div>
          <div class="legend"><span>低 margin</span><span class="alt">随机复核期望</span></div><div class="chart-callout">20% 预算捕获 56% 错误，约为随机复核的 2.8×；仍有 97 个错误未被复核覆盖。</div>
        </div>
      </section>
      <section class="section"><div class="section-head"><div><h2>预算明细</h2><p>同时展示捕获与残余，不只报“提升”。</p></div></div>
        <div class="table-wrap"><table><thead><tr><th>预算</th><th class="num">复核</th><th class="num">捕获错误</th><th class="num">错误召回</th><th class="num">残余</th><th class="num">富集</th></tr></thead><tbody>
          ${[['5%','62','42','19%','178','3.82×'],['10%','125','76','35%','144','3.45×'],['20%','250','123','56%','97','2.80×'],['30%','374','155','70%','65','2.34×'],['50%','624','194','88%','26','1.76×']].map(r => `<tr>${r.map((x,i)=>`<td data-label="${['预算','复核','捕获错误','错误召回','残余','富集'][i]}" class="${i ? 'num' : ''}">${x}</td>`).join('')}</tr>`).join('')}
        </tbody></table></div>
        <div class="method-note" style="margin-top:12px"><strong>不是临床分流建议</strong><p>排序只反映模型错误信号；未接入医生定义的临床严重度或后果权重。</p></div>
      </section>
    </div>`;
}

function casesPage() {
  const rows = [
    ['病例 0001','类别 07','类别 12','.91','.08','.11','预测错误；高置信错误'],
    ['病例 0002','类别 31','类别 14','.84','.29','.17','预测错误；优先复核候选'],
    ['病例 0003','类别 04','类别 09','.77','.46','.09','预测错误；高不确定性'],
    ['病例 0004','类别 18','类别 18','.64','.72','.05','高不确定性；类别边界不稳定'],
    ['病例 0005','类别 22','类别 21','.59','.81','.03','多版本预测不一致'],
    ['病例 0006','类别 45','类别 45','.88','.22','.31','未命中当前风险标签'],
    ['病例 0007','类别 12','类别 07','.82','.34','.16','预测错误；优先复核候选'],
  ];
  return `${auditHeader('cases')}
    <section class="section"><div class="toolbar">${field('病例范围', '优先复核候选')}${field('真实类别', '全部 59 类')}${field('预测类别', '全部 59 类')}${field('排序', '风险标签优先')}<div class="field"><label>会话序号</label><input class="input" placeholder="例如 0001"/></div>${button('导出筛选结果', 'small')}</div>
      <div class="table-wrap"><table><thead><tr><th style="width:12%">病例</th><th style="width:11%">真实类别</th><th style="width:11%">预测类别</th><th class="num">Confidence</th><th class="num">Entropy</th><th class="num">Margin</th><th style="width:31%">模型输出风险标签</th></tr></thead><tbody>
        ${rows.map(r => `<tr>${r.map((x,i)=>`<td data-label="${['病例','真实类别','预测类别','Confidence','Entropy','Margin','风险标签'][i]}" class="${i>=3&&i<=5?'num':''}">${i===0?`<a class="action-link" href="#">${x}</a>`:x}</td>`).join('')}</tr>`).join('')}
      </tbody></table></div>
      <p class="panel-note" style="margin-top:9px">默认不展示原始病例 ID、文件路径或服务器绝对路径；下载结果需显式确认。</p>
    </section>`;
}

function leakagePage() {
  const checks = [
    ['病例标识重复','未发现明显问题','0','68 个病例标识唯一','good'],
    ['病例跨 split','未发现明显问题','0','未发现同一病例跨 train/test','good'],
    ['标识或元数据包含类别名','发现可疑风险','3','3 个文件名片段需要人工核对','bad'],
    ['明显答案字段','未发现明显问题','0','未发现额外 diagnosis / ground_truth 字段','good'],
    ['元数据与标签近确定映射','当前无法评估','—','未选择可检查的元数据列','unknown'],
    ['患者级重叠','当前无法评估','—','缺少稳定患者标识','unknown'],
    ['图像级重叠','当前无法评估','—','缺少图像指纹与完整划分清单','unknown'],
    ['训练与测试流程隔离','当前无法评估','—','仅凭预测表无法确认训练流程','unknown'],
  ];
  return `${auditHeader('leakage')}
    <div class="split-3" style="margin-bottom:16px">
      <div class="panel panel-pad"><span class="kpi-label">未发现明显问题</span><strong class="kpi-value">3</strong><span class="kpi-meta">仅限当前表可观察证据</span></div>
      <div class="panel panel-pad"><span class="kpi-label">发现可疑风险</span><strong class="kpi-value" style="color:var(--danger)">1</strong><span class="kpi-meta">需要核对文件名来源</span></div>
      <div class="panel panel-pad"><span class="kpi-label">当前无法评估</span><strong class="kpi-value">4</strong><span class="kpi-meta">需补充患者、图像或训练信息</span></div>
    </div>
    <section class="section"><div class="section-head"><div><h2>数据完整性与标签捷径检查</h2><p>状态只允许三种，不显示“已确认无泄漏”。</p></div>${button('查看检查方法', 'small')}</div>
      <div class="check-list"><div class="check-row head"><span>检查项</span><span>状态</span><span>证据</span><span>说明</span></div>
        ${checks.map(([name,state,n,detail,kind]) => `<div class="check-row"><strong>${name}</strong><span>${status(state,kind)}</span><span class="num">${n}</span><span class="check-detail">${detail}</span></div>`).join('')}
      </div>
    </section>
    <div class="method-note"><strong>临床后果风险：尚未评估</strong><p>当前没有医生确认的严重度规则，本模块不会自行推断疾病或错误的临床后果。</p></div>`;
}

const renderers = { home: homePage, assets: assetsPage, readiness: readinessPage, research: researchPage, import: importPage, overview: overviewPage, risk: riskPage, cases: casesPage, leakage: leakagePage };
document.getElementById('app').innerHTML = shell((renderers[pageKey] || homePage)());
