import{_ as s,o as t,c as o,a as e,b as a}from"./app.547295de.js?v=20260626-configimport1";
const _={};
const c=e("h1",{id:"anima-fast-lora",tabindex:"-1"},[
  e("a",{class:"header-anchor",href:"#anima-fast-lora","aria-hidden":"true"},"#"),
  a(" Anima LoRA · Fast 模式")
],-1);
const n=e("p",null,"Anima 高速 LoRA 训练（进阶插件）。需单独安装 runtime，仅支持标准 LoRA。显存建议 16GB+，首次安装需下载数 GB 依赖。",-1);
const x=e("div",{class:"anima-fast-credit-root",innerHTML:"<p class=\"anima-fast-credit\">Fast 训练引擎来自开源项目 <a href=\"https://github.com/sorryhyun/anima_lora\" target=\"_blank\" rel=\"noopener noreferrer\">sorryhyun/anima_lora</a>。感谢原作者与社区的开发与分享；本页以可选插件形式集成，遵循各自开源许可。</p>"});
const d=e("div",{class:"anima-fast-doc-links-root",innerHTML:"<p class=\"anima-fast-doc-links\"><a href=\"https://github.com/wochenlong/lora-scripts-next/blob/main/docs/anima-fast.md\" target=\"_blank\" rel=\"noopener noreferrer\">Fast 模式训练教程</a>（安装、数据路径、故障排除） · <a href=\"/lora/sd3.html\">标准 Kohya 模式</a></p>"});
const g=e("div",{class:"anima-fast-guide-root",innerHTML:"<div class=\"anima-fast-guide-collapsible\">\n  <button type=\"button\" class=\"anima-fast-guide-toggle\" data-anima-fast-guide-toggle aria-expanded=\"false\">\n    <span class=\"anima-fast-guide-toggle__icon\" aria-hidden=\"true\">▸</span>\n    <span class=\"anima-fast-guide-toggle__label\">数据集路径说明（与 Kohya 不同）</span>\n  </button>\n  <div class=\"anima-fast-dataset-guide anima-fast-dataset-guide__body\" hidden>\n    <p>Fast 训练<strong>实际读取 resized 目录</strong>里的 bucket 预处理图，不是直接读原图。</p>\n    <ul>\n      <li><strong>训练图片目录</strong>：原图 + caption（如 <code>data/xxx/子文件夹/</code>）</li>\n      <li><strong>resized 目录</strong>：训练真正用到的 bucket PNG；<strong>留空</strong>时自动写入 <code>.cache/anima_fast/&lt;数据集路径&gt;/resized</code>（同一数据集可复用）</li>\n    </ul>\n    <p class=\"anima-fast-dataset-guide__highlight\"><strong>可以填同一路径吗？</strong>可以。若该目录已是 bucket 预处理后的 PNG + caption，两处可填<strong>相同路径</strong>。</p>\n    <p class=\"anima-fast-dataset-guide__note\">输出 / cache 目录不存在时会自动创建。左侧「cache_latents」等保持关闭，除非已完成完整 preprocess。</p>\n  </div>\n</div>"});
const m=e("div",{class:"anima-fast-install-panel",style:"display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0;"},[
  e("button",{"data-anima-fast-install":"",type:"button",class:"el-button el-button--primary is-plain"},[e("span",null,"安装 Fast 训练环境")]),
  e("span",{"data-anima-fast-status":"",style:"font-size:13px;opacity:.8;"},"检查中")
],-1);
const v=e("div",{"data-anima-fast-progress":"",hidden:"",style:"margin:10px 0 12px 0;padding:10px;border:1px solid var(--c-border);border-radius:6px;background:var(--c-bg-light);"},[
  e("div",{"data-anima-fast-progress-text":"",style:"font-size:13px;margin-bottom:8px;"},"准备安装"),
  e("div",{style:"height:8px;background:var(--c-border);border-radius:999px;overflow:hidden;"},[
    e("div",{"data-anima-fast-progress-bar":"",style:"width:0%;height:100%;background:linear-gradient(90deg,#6366f1,#22c55e);transition:width .25s ease;"})
  ]),
  e("div",{"data-anima-fast-progress-meta":"",style:"font-size:12px;opacity:.72;margin-top:6px;"},"0% · 预计剩余：正在估算")
],-1);
const f=e("pre",{"data-anima-fast-log":"",hidden:"",style:"max-height:260px;overflow:auto;margin:12px 0;padding:10px;border:1px solid var(--c-border);border-radius:6px;font-size:12px;line-height:1.45;white-space:pre-wrap;"},null,-1);
const l=[c,n,x,d,g,m,v,f];
function i(h,u){return t(),o("div",{class:"anima-fast-intro-wrap"},l)}
var p=s(_,[["render",i],["__file","anima-fast.html.vue"]]);
export{p as default};
