// static/js/webview-preload.js
const { ipcRenderer } = require('electron');


const newProto = navigator.__proto__;
delete newProto.webdriver;
navigator.__proto__ = newProto;

// 在文件最顶部添加
try {
  // 1. 移除 webdriver 属性
  Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
  });

  // 2. 伪装 Chrome 插件列表 (可选，增强伪装)
  if (!navigator.plugins || navigator.plugins.length === 0) {
     Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
     });
  }

} catch (e) {
  console.error('Failed to spoof navigator properties', e);
}

// 默认文本
let i18n = {
    translate: 'Translate',
    askAI: 'Ask AI',
    read: 'Read',
    copy: 'Copy',
    close: 'Close',
    loading: 'Generating...'
};

let toolbar = null;
let resultBox = null; // 新增：结果显示容器
let currentSelection = '';

// 初始化 DOM
function initToolbar() {
    if (document.getElementById('sap-ai-toolbar')) return;

    // 1. 注入样式
    const style = document.createElement('style');
    style.textContent = `
        #sap-ai-toolbar {
            position: fixed;
            z-index: 2147483647;
            background: #222;
            color: #fff;
            border-radius: 8px; /* 圆角稍微大一点 */
            padding: 6px;       /* 整体内边距增加 */
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            display: none;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
            user-select: none;
            flex-direction: column;
            align-items: flex-start;
            transition: opacity 0.2s, top 0.2s, left 0.2s; /* 增加位置平滑过渡 */
            pointer-events: auto;
            max-width: 380px; /* 稍微限制宽度，防止太宽 */
            width: auto;
            box-sizing: border-box; /* 关键：防止padding撑大 */
        }
        
        .sap-btn-row {
            display: flex;
            align-items: center;
            gap: 4px;
            width: 100%;
            padding: 0 2px;
            box-sizing: border-box;
        }

        #sap-ai-toolbar button {
            background: none;
            border: none;
            color: #ccc;
            padding: 5px 8px;
            cursor: pointer;
            display: inline-block;
            font-size: 12px;
            font-weight: 500;
            border-radius: 4px;
            transition: all 0.2s;
            outline: none;
            line-height: 1;
            white-space: nowrap;
        }
        #sap-ai-toolbar button:hover {
            background: rgba(255,255,255,0.2);
            color: #fff;
        }
        #sap-ai-toolbar .divider {
            width: 1px;
            background: #444;
            height: 14px;
            margin: 0 2px;
        }

        /* 结果显示区域 - 重点优化的部分 */
        #sap-ai-result {
            display: none;
            width: 100%;
            min-width: 240px; /* 最小宽度增加 */
            max-height: 300px;
            overflow-y: auto;
            
            /* 布局与边距 */
            box-sizing: border-box; /* 关键 */
            padding: 10px 12px;     /* 上下10px，左右12px，留足空间 */
            margin-top: 8px;
            
            /* 外观 */
            background-color: rgba(255, 255, 255, 0.05); /* 微微发亮的背景 */
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            
            /* 文字排版 */
            font-size: 13px;
            line-height: 1.6;
            color: #e0e0e0;
            
            /* 强制换行规则 */
            white-space: pre-wrap;       /* 保留换行 */
            word-wrap: break-word;       /* 旧版兼容 */
            overflow-wrap: break-word;   /* 标准换行 */
            word-break: break-word;      /* 防止长单词溢出 */
            
            user-select: text;
        }
        
        /* 滚动条美化 */
        #sap-ai-result::-webkit-scrollbar {
            width: 6px;
        }
        #sap-ai-result::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 3px;
        }
        #sap-ai-result::-webkit-scrollbar-track {
            background: transparent;
        }

        #sap-ai-result.active {
            display: block;
            animation: sap-fade-in 0.2s ease-out;
        }
        
        @keyframes sap-fade-in {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-close-btn {
            float: right;
            color: #888;
            cursor: pointer;
            margin-left: 8px;
            margin-bottom: 4px; /* 增加下边距，防止跟文字挤在一起 */
            font-size: 16px;
            line-height: 14px;
            font-weight: bold;
            transition: color 0.2s;
        }
        .result-close-btn:hover { color: #fff; }
    `;
    document.head.appendChild(style);

    // 2. 创建工具栏容器
    toolbar = document.createElement('div');
    toolbar.id = 'sap-ai-toolbar';
    document.body.appendChild(toolbar);
    
    // 渲染内部结构
    renderToolbarUI();

    // 3. 绑定 Webview 内部的交互事件
    bindEvents();
}

function renderToolbarUI() {
    if (!toolbar) return;
    
    toolbar.innerHTML = `
        <div class="sap-btn-row" id="sap-btns">
            <button id="ai-btn-trans">${i18n.translate}</button>
            <div class="divider"></div>
            <button id="ai-btn-ask">${i18n.askAI}</button>
            <div class="divider"></div>
            <button id="ai-btn-read">${i18n.read}</button>
            <div class="divider"></div>
            <button id="ai-btn-copy">${i18n.copy}</button>
        </div>
        <div id="sap-ai-result"></div>
    `;

    resultBox = document.getElementById('sap-ai-result');

    // 绑定点击
    document.getElementById('ai-btn-trans').onclick = () => sendAction('translate');
    document.getElementById('ai-btn-ask').onclick = () => sendAction('ask');
    document.getElementById('ai-btn-read').onclick = () => sendAction('read');
    document.getElementById('ai-btn-copy').onclick = () => {
        document.execCommand('copy');
        hideToolbar();
    };
}

function sendAction(action) {
    if (currentSelection) {
        ipcRenderer.sendToHost('ai-toolbar-action', { action, text: currentSelection });
        // 如果是翻译，保持工具栏显示，并进入加载状态
        if (action === 'translate') {
            showLoadingState();
        } else if (action === 'read') {
            // 朗读也可以保持显示，或者隐藏，看你喜好
            // hideToolbar(); 
        } else {
            // Ask AI 会去侧边栏，这里可以隐藏
            hideToolbar();
        }
    }
}

function showLoadingState() {
    if (!resultBox) return;
    resultBox.classList.add('active');
    resultBox.innerHTML = `<span style="color:#888;">${i18n.loading}...</span>`;
}

function hideToolbar() {
    if (toolbar) {
        toolbar.style.display = 'none';
        // 重置结果框
        if (resultBox) {
            resultBox.classList.remove('active');
            resultBox.innerHTML = '';
        }
    }
}

function bindEvents() {
    document.addEventListener('mouseup', (e) => {
        setTimeout(() => {
            const sel = window.getSelection();
            const text = sel.toString().trim();

            if (toolbar && toolbar.contains(e.target)) return;

            if (text && text.length > 0) {
                currentSelection = text;
                const range = sel.getRangeAt(0);
                const rect = range.getBoundingClientRect();

                if (toolbar) {
                    // 如果结果框是打开的，说明正在看上一次的结果，不要因为重新划词而乱动位置，
                    // 除非用户点击了页面其他地方（上面的 if toolbar.contains 已经处理了）
                    // 但这里策略是：新划词 = 重置工具栏
                    if (resultBox && resultBox.classList.contains('active')) {
                        // 如果正在显示结果，这时候划新词，可以先隐藏旧结果
                        resultBox.classList.remove('active');
                    }

                    let top = rect.top - 40;
                    let left = rect.left + (rect.width / 2) - (toolbar.offsetWidth / 2);

                    if (top < 10) top = rect.bottom + 10;
                    if (left < 10) left = 10;
                    if (left + toolbar.offsetWidth > window.innerWidth) left = window.innerWidth - toolbar.offsetWidth - 10;

                    toolbar.style.top = top + 'px';
                    toolbar.style.left = left + 'px';
                    toolbar.style.display = 'flex';
                }
            } else {
                hideToolbar();
            }
        }, 100);
    });

    document.addEventListener('mousedown', (e) => {
        if (toolbar && !toolbar.contains(e.target)) {
            hideToolbar();
        }
    });

    document.addEventListener('scroll', hideToolbar, { passive: true, capture: true });
}

window.addEventListener('DOMContentLoaded', () => {
    initToolbar();
});

ipcRenderer.on('set-i18n', (event, data) => {
    if (data) {
        i18n = { ...i18n, ...data };
        renderToolbarUI();
    }
});

// --- 新增：接收流式数据的监听器 ---
ipcRenderer.on('ai-stream-start', () => {
    if (resultBox) {
        resultBox.innerHTML = ''; // 清空加载中提示
        // 添加一个关闭按钮
        const closeBtn = document.createElement('span');
        closeBtn.className = 'result-close-btn';
        closeBtn.innerHTML = '&times;';
        closeBtn.onclick = (e) => { e.stopPropagation(); hideToolbar(); };
        resultBox.appendChild(closeBtn);
        
        // 内容容器
        const contentSpan = document.createElement('span');
        contentSpan.id = 'sap-stream-content';
        resultBox.appendChild(contentSpan);
    }
});

ipcRenderer.on('ai-stream-chunk', (event, text) => {
    const contentSpan = document.getElementById('sap-stream-content');
    if (contentSpan) {
        // 简单处理换行
        contentSpan.innerText += text;
        // 自动滚动到底部
        if (resultBox) resultBox.scrollTop = resultBox.scrollHeight;
    }
});

ipcRenderer.on('ai-stream-end', () => {
    // 结束时的处理，比如光标闪烁停止等，这里暂时不用做特别处理
});