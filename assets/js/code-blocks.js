document.addEventListener('DOMContentLoaded', function () {
  const preBlocks = document.querySelectorAll('.tutorial-page pre');

  preBlocks.forEach(function (pre) {
    const code = pre.querySelector('code');
    if (!code) return;

    // Detect language from class (e.g. "language-python")
    const langMatch = code.className.match(/language-(\w+)/);
    const lang = langMatch ? langMatch[1] : '';

    // Create wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper';
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    // Create header bar
    const header = document.createElement('div');
    header.className = 'code-header';

    const langSpan = document.createElement('span');
    langSpan.className = 'code-lang';
    langSpan.textContent = lang || 'code';
    header.appendChild(langSpan);

    // Copy button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>' +
      '<span>复制</span>';
    copyBtn.setAttribute('aria-label', '复制代码');
    copyBtn.addEventListener('click', function () {
      const text = code.textContent;
      navigator.clipboard.writeText(text).then(function () {
        copyBtn.classList.add('copied');
        copyBtn.querySelector('span').textContent = '已复制';
        setTimeout(function () {
          copyBtn.classList.remove('copied');
          copyBtn.querySelector('span').textContent = '复制';
        }, 2000);
      }).catch(function () {
        // Fallback for older browsers
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        copyBtn.classList.add('copied');
        copyBtn.querySelector('span').textContent = '已复制';
        setTimeout(function () {
          copyBtn.classList.remove('copied');
          copyBtn.querySelector('span').textContent = '复制';
        }, 2000);
      });
    });
    header.appendChild(copyBtn);

    wrapper.insertBefore(header, pre);
  });
});
