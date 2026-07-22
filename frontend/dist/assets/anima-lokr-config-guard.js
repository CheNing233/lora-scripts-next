(function () {
  "use strict";

  const LYCO_MODULE_RE = /\bnetwork_module\s*=\s*["']lycoris\.kohya["']/i;
  const NETWORK_ARGS_RE = /(^[ \t]*network_args\s*=\s*\[)([\s\S]*?)(^[ \t]*\])/gim;
  const QUOTED_ITEM_RE = /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g;
  const INVALID_ARG_RE = /^[^=\s]+\s*=\s*(?:undefined|null|nan)$/i;

  function sanitizeLycorisToml(value) {
    if (typeof value !== "string" || !LYCO_MODULE_RE.test(value)) {
      return value;
    }

    return value.replace(NETWORK_ARGS_RE, function (whole, opening, body, closing) {
      const items = body.match(QUOTED_ITEM_RE);
      if (!items) {
        return whole;
      }

      const kept = items.filter(function (quoted) {
        const raw = quoted.slice(1, -1).trim();
        return !INVALID_ARG_RE.test(raw);
      });
      if (kept.length === items.length) {
        return whole;
      }
      if (kept.length === 0) {
        return opening + closing;
      }

      const indentMatch = body.match(/\r?\n([ \t]*)["']/);
      const indent = indentMatch ? indentMatch[1] : "  ";
      const newline = value.includes("\r\n") ? "\r\n" : "\n";
      return opening + newline + indent + kept.join("," + newline + indent) + newline + closing;
    });
  }

  window.mikazukiSanitizeLycorisTomlText = sanitizeLycorisToml;

  // The vendored layout creates the downloaded TOML with new Blob([text]).
  // Restrict the override to this page and only transform string parts that
  // contain a LyCORIS config; every other Blob remains byte-for-byte unchanged.
  const NativeBlob = window.Blob;
  if (typeof NativeBlob === "function") {
    class SanitizedConfigBlob extends NativeBlob {
      constructor(parts, options) {
        const safeParts = Array.isArray(parts)
          ? parts.map(function (part) {
              return typeof part === "string" ? sanitizeLycorisToml(part) : part;
            })
          : parts;
        super(safeParts, options);
      }
    }
    window.Blob = SanitizedConfigBlob;
  }

  function sanitizePreviewTextNode(node) {
    if (
      node.nodeType !== Node.TEXT_NODE ||
      !node.parentElement ||
      !node.parentElement.closest(".params-section")
    ) {
      return;
    }
    const cleaned = sanitizeLycorisToml(node.nodeValue);
    if (cleaned !== node.nodeValue) {
      node.nodeValue = cleaned;
    }
  }

  function sanitizePreviewTree(root) {
    if (!root) {
      return;
    }
    if (root.nodeType === Node.TEXT_NODE) {
      sanitizePreviewTextNode(root);
      return;
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      sanitizePreviewTextNode(node);
    }
  }

  function installPreviewGuard() {
    const preview = document.querySelector(".params-section");
    if (!preview) {
      return false;
    }
    sanitizePreviewTree(preview);
    new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.type === "characterData") {
          sanitizePreviewTextNode(mutation.target);
          return;
        }
        mutation.addedNodes.forEach(sanitizePreviewTree);
      });
    }).observe(preview, {
      childList: true,
      characterData: true,
      subtree: true,
    });
    return true;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installPreviewGuard, { once: true });
  } else {
    installPreviewGuard();
  }
})();
