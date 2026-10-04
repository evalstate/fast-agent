/* @ds-bundle: {"format":4,"namespace":"FastAgentDesignSystem_3898e4","components":[{"name":"Button","sourcePath":"components/actions/Button.jsx"},{"name":"Link","sourcePath":"components/actions/Link.jsx"},{"name":"Burst","sourcePath":"components/display/Burst.jsx"},{"name":"Card","sourcePath":"components/display/Card.jsx"},{"name":"CodeBlock","sourcePath":"components/display/CodeBlock.jsx"},{"name":"FootnoteMark","sourcePath":"components/display/FootnoteMark.jsx"},{"name":"Sticker","sourcePath":"components/display/Sticker.jsx"},{"name":"Table","sourcePath":"components/display/Table.jsx"},{"name":"Tag","sourcePath":"components/display/Tag.jsx"},{"name":"Wordmark","sourcePath":"components/display/Wordmark.jsx"},{"name":"Dialog","sourcePath":"components/feedback/Dialog.jsx"},{"name":"Loader","sourcePath":"components/feedback/Loader.jsx"},{"name":"Toast","sourcePath":"components/feedback/Toast.jsx"},{"name":"Tooltip","sourcePath":"components/feedback/Tooltip.jsx"},{"name":"Checkbox","sourcePath":"components/forms/Checkbox.jsx"},{"name":"Input","sourcePath":"components/forms/Input.jsx"},{"name":"Radio","sourcePath":"components/forms/Radio.jsx"},{"name":"Select","sourcePath":"components/forms/Select.jsx"},{"name":"Switch","sourcePath":"components/forms/Switch.jsx"},{"name":"Tabs","sourcePath":"components/navigation/Tabs.jsx"}],"sourceHashes":{"components/actions/Button.jsx":"da49122f379e","components/actions/Link.jsx":"0f1df0538876","components/display/Burst.jsx":"20e61fe1823a","components/display/Card.jsx":"8e97990f75c5","components/display/CodeBlock.jsx":"b0f020e41e53","components/display/FootnoteMark.jsx":"ce3ffe7a1134","components/display/Sticker.jsx":"a1a0eee3092c","components/display/Table.jsx":"1f95e570ba76","components/display/Tag.jsx":"39a501580a67","components/display/Wordmark.jsx":"a638529fcd1c","components/feedback/Dialog.jsx":"6ebc6152a195","components/feedback/Loader.jsx":"63589ae117a8","components/feedback/Toast.jsx":"1d27f8b12291","components/feedback/Tooltip.jsx":"a18f46753eaa","components/forms/Checkbox.jsx":"6df01a1992db","components/forms/Input.jsx":"5223f23056bd","components/forms/Radio.jsx":"da737ce69060","components/forms/Select.jsx":"17bcf41ba7a8","components/forms/Switch.jsx":"991319964aa0","components/navigation/Tabs.jsx":"44ae68c9401c","ui_kits/website/BenchmarkChart.jsx":"489c8eb8155f","ui_kits/website/BenchmarksPage.jsx":"8cccd242bfa8","ui_kits/website/DocsPage.jsx":"8adfec13e514","ui_kits/website/HomePage.jsx":"3ea8b9d090ef","ui_kits/website/Shell.jsx":"a587bdd9a593","ui_kits/website/data.js":"b524eee36c47"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.FastAgentDesignSystem_3898e4 = window.FastAgentDesignSystem_3898e4 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/actions/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const SIZES = {
  sm: {
    font: 13,
    pad: '8px 14px'
  },
  md: {
    font: 15,
    pad: '11px 18px'
  },
  lg: {
    font: 17,
    pad: '14px 24px'
  }
};
function Button({
  variant = 'primary',
  size = 'md',
  disabled = false,
  children,
  onClick,
  type = 'button',
  style,
  ...rest
}) {
  const [hover, setHover] = React.useState(false);
  const [down, setDown] = React.useState(false);
  const [focused, setFocused] = React.useState(false);
  const s = SIZES[size] || SIZES.md;
  const live = !disabled;
  const base = {
    fontFamily: 'var(--font-read)',
    fontWeight: 700,
    fontSize: s.font,
    lineHeight: 1.2,
    borderRadius: 'var(--radius-md)',
    cursor: live ? 'pointer' : 'not-allowed',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    whiteSpace: 'nowrap',
    transition: 'transform var(--dur-hover) var(--ease-ui), box-shadow var(--dur-hover) var(--ease-ui), background var(--dur-hover) var(--ease-ui)',
    outline: focused ? '3px solid var(--focus-ring)' : 'none',
    outlineOffset: 2,
    opacity: disabled ? 0.45 : 1
  };
  let v;
  if (variant === 'primary') {
    const ledge = !live ? 4 : down ? 0 : hover ? 2 : 4;
    v = {
      background: 'var(--petrol)',
      color: 'var(--ivory)',
      border: 0,
      padding: s.pad,
      boxShadow: '0 ' + ledge + 'px 0 var(--amber)',
      transform: 'translateY(' + (4 - ledge) + 'px)'
    };
  } else if (variant === 'secondary') {
    const [y, x] = s.pad.split(' ').map(n => parseInt(n) - 2);
    v = {
      background: live && hover ? 'var(--amber)' : 'transparent',
      color: 'var(--petrol)',
      border: 'var(--border)',
      padding: y + 'px ' + x + 'px'
    };
  } else {
    v = {
      background: 'transparent',
      color: live && hover ? 'var(--teal)' : 'var(--petrol)',
      border: 0,
      padding: s.pad,
      boxShadow: 'inset 0 -' + (live && hover ? 4 : 2) + 'px 0 var(--amber)',
      borderRadius: 0,
      paddingLeft: 0,
      paddingRight: 0,
      paddingBottom: 4,
      paddingTop: 4
    };
  }
  return /*#__PURE__*/React.createElement("button", _extends({
    type: type,
    disabled: disabled,
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => {
      setHover(false);
      setDown(false);
    },
    onMouseDown: () => setDown(true),
    onMouseUp: () => setDown(false),
    onFocus: e => setFocused(e.target.matches(':focus-visible')),
    onBlur: () => setFocused(false),
    style: {
      ...base,
      ...v,
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/actions/Button.jsx", error: String((e && e.message) || e) }); }

// components/actions/Link.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Link({
  href = '#',
  children,
  style,
  ...rest
}) {
  const [hover, setHover] = React.useState(false);
  return /*#__PURE__*/React.createElement("a", _extends({
    href: href,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      fontFamily: 'inherit',
      fontWeight: 600,
      color: hover ? 'var(--link-hover)' : 'var(--link)',
      textDecoration: 'none',
      boxShadow: 'inset 0 -' + (hover ? 4 : 2) + 'px 0 var(--link-underline)',
      transition: 'box-shadow var(--dur-hover) var(--ease-ui), color var(--dur-hover) var(--ease-ui)',
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Link });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/actions/Link.jsx", error: String((e && e.message) || e) }); }

// components/display/Burst.jsx
try { (() => {
function Burst({
  src = 'assets/burst-capsule-amber.svg',
  size = 140,
  tilt = -8,
  children,
  pop = true,
  textColor = 'var(--petrol)',
  style
}) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      '--tilt': tilt + 'deg',
      position: 'relative',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: size,
      height: size,
      transform: 'rotate(' + tilt + 'deg)',
      animation: pop ? 'fa-pop var(--dur-pop) var(--ease-pop) both' : 'none',
      ...style
    }
  }, /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: "",
    width: size,
    height: size,
    style: {
      position: 'absolute',
      inset: 0,
      display: 'block'
    }
  }), children && /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'relative',
      fontFamily: 'var(--font-shout)',
      fontSize: Math.round(size * 0.16),
      lineHeight: 1,
      color: textColor,
      textAlign: 'center',
      maxWidth: '62%'
    }
  }, children));
}
Object.assign(__ds_scope, { Burst });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/Burst.jsx", error: String((e && e.message) || e) }); }

// components/display/Card.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const V = {
  paper: {
    background: 'var(--paper)',
    border: 'var(--border)',
    color: 'var(--petrol)'
  },
  ivory: {
    background: 'var(--ivory)',
    border: 'var(--border)',
    color: 'var(--petrol)'
  },
  inset: {
    background: 'var(--ivory-deep)',
    border: '2px solid transparent',
    color: 'var(--petrol)'
  },
  inverse: {
    background: 'var(--petrol)',
    border: '2px solid var(--petrol)',
    color: 'var(--ivory)'
  }
};
function Card({
  variant = 'paper',
  padding = 24,
  children,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      ...V[variant],
      borderRadius: 'var(--radius-lg)',
      padding,
      boxSizing: 'border-box',
      fontFamily: 'var(--font-read)',
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Card });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/Card.jsx", error: String((e && e.message) || e) }); }

// components/display/CodeBlock.jsx
try { (() => {
function CodeBlock({
  lines = [],
  prompt = true,
  title,
  style
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--petrol)',
      color: 'var(--ivory)',
      borderRadius: 'var(--radius-lg)',
      fontFamily: 'var(--font-code)',
      fontSize: 14,
      lineHeight: 1.7,
      overflow: 'hidden',
      ...style
    }
  }, title && /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '8px 16px',
      borderBottom: '2px solid var(--petrol-2)',
      fontFamily: 'var(--font-read)',
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: 'var(--track-label)',
      textTransform: 'uppercase',
      color: 'var(--amber)'
    }
  }, title), /*#__PURE__*/React.createElement("pre", {
    style: {
      margin: 0,
      padding: '14px 16px',
      fontFamily: 'inherit',
      whiteSpace: 'pre-wrap'
    }
  }, lines.map((l, i) => {
    const cmd = typeof l === 'string' ? prompt : l.cmd;
    const t = typeof l === 'string' ? l : l.text;
    return /*#__PURE__*/React.createElement("div", {
      key: i
    }, cmd && /*#__PURE__*/React.createElement("span", {
      style: {
        color: 'var(--amber)',
        marginRight: 10
      }
    }, "\u276F"), /*#__PURE__*/React.createElement("span", {
      style: {
        opacity: cmd ? 1 : 0.78
      }
    }, t));
  })));
}
Object.assign(__ds_scope, { CodeBlock });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/CodeBlock.jsx", error: String((e && e.message) || e) }); }

// components/display/FootnoteMark.jsx
try { (() => {
function FootnoteMark({
  n,
  href,
  sprite = 'assets/sprite.svg',
  size = 14
}) {
  const mark = /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 2,
      verticalAlign: 'super',
      fontFamily: 'var(--font-read)',
      fontSize: 11,
      fontWeight: 800,
      color: 'var(--petrol)',
      lineHeight: 1
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: size,
    height: size,
    viewBox: "0 0 100 100",
    style: {
      color: 'var(--amber)',
      display: 'block'
    },
    "aria-hidden": "true"
  }, /*#__PURE__*/React.createElement("use", {
    href: sprite + '#fa-burst'
  })), n);
  return href ? /*#__PURE__*/React.createElement("a", {
    href: href,
    style: {
      textDecoration: 'none'
    },
    "aria-label": 'Footnote ' + n
  }, mark) : mark;
}
Object.assign(__ds_scope, { FootnoteMark });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/FootnoteMark.jsx", error: String((e && e.message) || e) }); }

// components/display/Sticker.jsx
try { (() => {
function Sticker({
  children,
  tilt = -8,
  color = 'amber',
  pop = true,
  style
}) {
  const bg = color === 'orange' ? 'var(--orange)' : color === 'ivory' ? 'var(--ivory)' : 'var(--amber)';
  return /*#__PURE__*/React.createElement("span", {
    style: {
      '--tilt': tilt + 'deg',
      display: 'inline-block',
      transform: 'rotate(' + tilt + 'deg)',
      animation: pop ? 'fa-pop var(--dur-pop) var(--ease-pop) both' : 'none',
      background: bg,
      color: 'var(--petrol)',
      border: 'var(--border)',
      borderRadius: 'var(--radius-md)',
      boxShadow: 'var(--shadow-offset-sm)',
      fontFamily: 'var(--font-shout)',
      fontSize: 18,
      lineHeight: 1,
      padding: '8px 14px',
      whiteSpace: 'nowrap',
      ...style
    }
  }, children);
}
Object.assign(__ds_scope, { Sticker });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/Sticker.jsx", error: String((e && e.message) || e) }); }

// components/display/Table.jsx
try { (() => {
function Table({
  columns = [],
  rows = [],
  highlight,
  caption,
  footnote
}) {
  const cell = {
    padding: '10px 14px',
    borderBottom: '2px solid var(--line)',
    textAlign: 'left'
  };
  return /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: 'var(--font-read)',
      color: 'var(--petrol)'
    }
  }, /*#__PURE__*/React.createElement("table", {
    style: {
      width: '100%',
      borderCollapse: 'collapse',
      fontSize: 15,
      border: 'var(--border)',
      borderRadius: 'var(--radius-lg)',
      borderSpacing: 0,
      overflow: 'hidden'
    }
  }, caption && /*#__PURE__*/React.createElement("caption", {
    style: {
      textAlign: 'left',
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: 'var(--track-label)',
      textTransform: 'uppercase',
      paddingBottom: 8
    }
  }, caption), /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, columns.map(c => /*#__PURE__*/React.createElement("th", {
    key: c.key,
    style: {
      ...cell,
      textAlign: c.numeric ? 'right' : 'left',
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: 'var(--track-label)',
      textTransform: 'uppercase',
      borderBottom: 'var(--border)',
      background: 'var(--ivory-deep)'
    }
  }, c.label)))), /*#__PURE__*/React.createElement("tbody", null, rows.map((r, i) => {
    const hi = highlight != null && (typeof highlight === 'function' ? highlight(r, i) : highlight === i);
    return /*#__PURE__*/React.createElement("tr", {
      key: i,
      style: {
        background: hi ? 'var(--amber)' : 'var(--paper)'
      }
    }, columns.map(c => /*#__PURE__*/React.createElement("td", {
      key: c.key,
      style: {
        ...cell,
        borderBottom: i === rows.length - 1 ? 0 : cell.borderBottom,
        textAlign: c.numeric ? 'right' : 'left',
        fontWeight: c.numeric ? 900 : hi ? 800 : 500,
        fontVariantNumeric: 'tabular-nums'
      }
    }, r[c.key])));
  }))), footnote && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      lineHeight: 1.5,
      color: 'var(--text-muted)',
      marginTop: 8
    }
  }, footnote));
}
Object.assign(__ds_scope, { Table });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/Table.jsx", error: String((e && e.message) || e) }); }

// components/display/Tag.jsx
try { (() => {
const V = {
  outline: {
    background: 'transparent',
    color: 'var(--petrol)',
    border: 'var(--border)'
  },
  accent: {
    background: 'var(--amber)',
    color: 'var(--petrol)',
    border: '2px solid var(--amber)'
  },
  inverse: {
    background: 'var(--petrol)',
    color: 'var(--ivory)',
    border: '2px solid var(--petrol)'
  },
  muted: {
    background: 'var(--ivory-deep)',
    color: 'var(--petrol)',
    border: '2px solid var(--ivory-deep)'
  }
};
function Tag({
  variant = 'outline',
  children,
  style
}) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      ...V[variant],
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      fontFamily: 'var(--font-read)',
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: 'var(--track-label)',
      textTransform: 'uppercase',
      lineHeight: 1,
      padding: '5px 8px',
      borderRadius: 'var(--radius-sm)',
      whiteSpace: 'nowrap',
      ...style
    }
  }, children);
}
Object.assign(__ds_scope, { Tag });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/Tag.jsx", error: String((e && e.message) || e) }); }

// components/display/Wordmark.jsx
try { (() => {
function Wordmark({
  size = 32,
  inverse = false,
  icon = true,
  assetBase = 'assets/'
}) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: Math.round(size * 0.35),
      color: inverse ? 'var(--ivory)' : 'var(--petrol)'
    }
  }, icon && /*#__PURE__*/React.createElement("img", {
    src: assetBase + (inverse ? 'icon-tile-amber.svg' : 'icon-tile.svg'),
    width: size,
    height: size,
    alt: "",
    style: {
      display: 'block'
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-voice)',
      fontWeight: 900,
      fontVariationSettings: 'var(--voice-settings)',
      letterSpacing: 'var(--track-voice)',
      fontSize: size,
      lineHeight: 1,
      whiteSpace: 'nowrap'
    }
  }, "fast-agent"));
}
Object.assign(__ds_scope, { Wordmark });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/display/Wordmark.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Dialog.jsx
try { (() => {
function Dialog({
  open = true,
  title,
  children,
  actions,
  onClose,
  inline = false,
  width = 460
}) {
  if (!open) return null;
  const panel = /*#__PURE__*/React.createElement("div", {
    role: "dialog",
    "aria-modal": !inline,
    style: {
      background: 'var(--paper)',
      color: 'var(--petrol)',
      border: 'var(--border)',
      borderRadius: 'var(--radius-lg)',
      padding: 24,
      width,
      maxWidth: '100%',
      boxSizing: 'border-box',
      fontFamily: 'var(--font-read)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      alignItems: 'flex-start',
      gap: 16
    }
  }, title && /*#__PURE__*/React.createElement("h2", {
    style: {
      margin: 0,
      flex: 1,
      fontFamily: 'var(--font-voice)',
      fontWeight: 900,
      fontVariationSettings: 'var(--voice-settings)',
      letterSpacing: 'var(--track-voice)',
      fontSize: 26,
      lineHeight: 1.1
    }
  }, title), onClose && /*#__PURE__*/React.createElement("button", {
    onClick: onClose,
    "aria-label": "Close",
    style: {
      background: 'none',
      border: 0,
      fontFamily: 'var(--font-code)',
      fontSize: 20,
      lineHeight: 1,
      color: 'var(--petrol)',
      cursor: 'pointer',
      padding: 0
    }
  }, "\xD7")), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 15,
      lineHeight: 1.6,
      marginTop: 10
    }
  }, children), actions && /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'flex-end',
      gap: 12,
      marginTop: 22
    }
  }, actions));
  if (inline) return panel;
  return /*#__PURE__*/React.createElement("div", {
    onClick: e => {
      if (e.target === e.currentTarget && onClose) onClose();
    },
    style: {
      position: 'fixed',
      inset: 0,
      background: 'rgba(8, 44, 52, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
      zIndex: 100
    }
  }, panel);
}
Object.assign(__ds_scope, { Dialog });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Dialog.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Loader.jsx
try { (() => {
function Loader({
  kind = 'ratchet',
  size = 40,
  assetBase = 'assets/',
  label
}) {
  let el;
  if (kind === 'caret') el = /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-code)',
      fontSize: size * 0.55,
      color: 'var(--petrol)',
      display: 'inline-flex',
      alignItems: 'center'
    }
  }, "\u276F", /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-block',
      width: size * 0.28,
      height: size * 0.55,
      background: 'var(--amber)',
      marginLeft: size * 0.15,
      animation: 'fa-prompt 1s steps(1) infinite'
    }
  }));else if (kind === 'chase') el = /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-flex',
      gap: 2
    }
  }, [0, 0.18, 0.36].map(d => /*#__PURE__*/React.createElement("img", {
    key: d,
    src: assetBase + 'mark-chevron.svg',
    width: size * 0.6,
    alt: "",
    style: {
      animation: 'fa-chevrons 1.2s var(--ease-ui) ' + d + 's infinite'
    }
  })));else el = /*#__PURE__*/React.createElement("img", {
    src: assetBase + 'burst-capsule-amber.svg',
    width: size,
    height: size,
    alt: "",
    style: {
      display: 'block',
      animation: 'fa-ratchet 1.1s var(--ease-ratchet) infinite'
    }
  });
  return /*#__PURE__*/React.createElement("span", {
    role: "status",
    "aria-label": label || 'Loading',
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      fontFamily: 'var(--font-read)',
      fontSize: 14,
      fontWeight: 600,
      color: 'var(--petrol)'
    }
  }, el, label);
}
Object.assign(__ds_scope, { Loader });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Loader.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Toast.jsx
try { (() => {
function Toast({
  children,
  action,
  onAction,
  onClose,
  tone = 'default'
}) {
  return /*#__PURE__*/React.createElement("div", {
    role: "status",
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 16,
      background: 'var(--petrol)',
      color: 'var(--ivory)',
      borderRadius: 'var(--radius-md)',
      padding: '12px 16px',
      fontFamily: 'var(--font-read)',
      fontSize: 14,
      fontWeight: 600,
      lineHeight: 1.4,
      maxWidth: 440,
      boxSizing: 'border-box',
      borderLeft: 0
    }
  }, tone === 'danger' && /*#__PURE__*/React.createElement("span", {
    style: {
      width: 10,
      height: 10,
      borderRadius: 2,
      background: 'var(--orange)',
      flex: 'none'
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      flex: 1
    }
  }, children), action && /*#__PURE__*/React.createElement("button", {
    onClick: onAction,
    style: {
      background: 'none',
      border: 0,
      padding: '2px 0',
      color: 'var(--ivory)',
      fontFamily: 'inherit',
      fontSize: 14,
      fontWeight: 800,
      cursor: 'pointer',
      boxShadow: 'inset 0 -2px 0 var(--amber)'
    }
  }, action), onClose && /*#__PURE__*/React.createElement("button", {
    onClick: onClose,
    "aria-label": "Dismiss",
    style: {
      background: 'none',
      border: 0,
      color: 'var(--ivory)',
      fontFamily: 'var(--font-code)',
      fontSize: 16,
      cursor: 'pointer',
      padding: 0,
      lineHeight: 1
    }
  }, "\xD7"));
}
Object.assign(__ds_scope, { Toast });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Toast.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Tooltip.jsx
try { (() => {
function Tooltip({
  content,
  children,
  placement = 'top',
  open
}) {
  const [hover, setHover] = React.useState(false);
  const show = open ?? hover;
  const pos = placement === 'bottom' ? {
    top: '100%',
    marginTop: 8
  } : {
    bottom: '100%',
    marginBottom: 8
  };
  return /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'relative',
      display: 'inline-flex'
    },
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    onFocus: () => setHover(true),
    onBlur: () => setHover(false)
  }, children, show && /*#__PURE__*/React.createElement("span", {
    role: "tooltip",
    style: {
      position: 'absolute',
      left: '50%',
      transform: 'translateX(-50%)',
      ...pos,
      background: 'var(--petrol)',
      color: 'var(--ivory)',
      fontFamily: 'var(--font-read)',
      fontSize: 13,
      fontWeight: 500,
      lineHeight: 1.4,
      padding: '6px 10px',
      borderRadius: 'var(--radius-sm)',
      whiteSpace: 'nowrap',
      zIndex: 10,
      pointerEvents: 'none'
    }
  }, content));
}
Object.assign(__ds_scope, { Tooltip });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Tooltip.jsx", error: String((e && e.message) || e) }); }

// components/forms/Checkbox.jsx
try { (() => {
function Checkbox({
  checked,
  defaultChecked = false,
  onChange,
  label,
  disabled = false
}) {
  const [inner, setInner] = React.useState(defaultChecked);
  const on = checked ?? inner;
  const [focused, setFocused] = React.useState(false);
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      fontFamily: 'var(--font-read)',
      fontSize: 15,
      fontWeight: 600,
      color: 'var(--petrol)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.45 : 1
    }
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: on,
    disabled: disabled,
    onChange: e => {
      setInner(e.target.checked);
      onChange && onChange(e);
    },
    onFocus: e => setFocused(e.target.matches(':focus-visible')),
    onBlur: () => setFocused(false),
    style: {
      position: 'absolute',
      opacity: 0,
      width: 1,
      height: 1
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      width: 20,
      height: 20,
      boxSizing: 'border-box',
      border: 'var(--border)',
      borderRadius: 5,
      background: on ? 'var(--amber)' : 'var(--paper)',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      transition: 'background var(--dur-hover) var(--ease-ui)',
      outline: focused ? '3px solid var(--focus-ring)' : 'none',
      outlineOffset: 2,
      flex: 'none'
    }
  }, on && /*#__PURE__*/React.createElement("span", {
    style: {
      width: 5,
      height: 10,
      borderRight: '2.5px solid var(--petrol)',
      borderBottom: '2.5px solid var(--petrol)',
      transform: 'translateY(-1px) rotate(45deg)'
    }
  })), label);
}
Object.assign(__ds_scope, { Checkbox });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Checkbox.jsx", error: String((e && e.message) || e) }); }

// components/forms/Input.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function Input({
  label,
  hint,
  error,
  disabled = false,
  mono = false,
  id,
  style,
  ...rest
}) {
  const [focused, setFocused] = React.useState(false);
  const iid = id || React.useId();
  return /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: 'var(--font-read)',
      color: 'var(--petrol)',
      ...style
    }
  }, label && /*#__PURE__*/React.createElement("label", {
    htmlFor: iid,
    style: {
      display: 'block',
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: 'var(--track-label)',
      textTransform: 'uppercase',
      marginBottom: 6,
      color: 'var(--petrol)'
    }
  }, label), /*#__PURE__*/React.createElement("input", _extends({
    id: iid,
    disabled: disabled,
    onFocus: () => setFocused(true),
    onBlur: () => setFocused(false),
    style: {
      width: '100%',
      boxSizing: 'border-box',
      fontFamily: mono ? 'var(--font-code)' : 'var(--font-read)',
      fontSize: 15,
      fontWeight: 500,
      color: 'var(--petrol)',
      background: disabled ? 'var(--ivory-deep)' : 'var(--paper)',
      border: '2px solid ' + (error ? 'var(--orange)' : 'var(--petrol)'),
      borderRadius: 'var(--radius-sm)',
      padding: '10px 12px',
      outline: focused ? '3px solid var(--focus-ring)' : 'none',
      outlineOffset: 2,
      opacity: disabled ? 0.6 : 1
    }
  }, rest)), (error || hint) && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      lineHeight: 1.5,
      marginTop: 6,
      color: error ? 'var(--petrol)' : 'var(--text-muted)',
      fontWeight: error ? 700 : 400
    }
  }, error ? /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-block',
      width: 8,
      height: 8,
      background: 'var(--orange)',
      borderRadius: 2,
      marginRight: 6
    }
  }) : null, error || hint));
}
Object.assign(__ds_scope, { Input });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Input.jsx", error: String((e && e.message) || e) }); }

// components/forms/Radio.jsx
try { (() => {
function Radio({
  name,
  options = [],
  value,
  defaultValue,
  onChange,
  disabled = false,
  direction = 'column'
}) {
  const [inner, setInner] = React.useState(defaultValue ?? (options[0] && (options[0].value ?? options[0])));
  const cur = value ?? inner;
  return /*#__PURE__*/React.createElement("div", {
    role: "radiogroup",
    style: {
      display: 'flex',
      flexDirection: direction,
      gap: direction === 'row' ? 20 : 10,
      fontFamily: 'var(--font-read)',
      color: 'var(--petrol)'
    }
  }, options.map(o => {
    const v = o.value ?? o;
    const l = o.label ?? o;
    const on = cur === v;
    return /*#__PURE__*/React.createElement("label", {
      key: v,
      style: {
        display: 'inline-flex',
        alignItems: 'center',
        gap: 10,
        fontSize: 15,
        fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.45 : 1
      }
    }, /*#__PURE__*/React.createElement("input", {
      type: "radio",
      name: name,
      value: v,
      checked: on,
      disabled: disabled,
      onChange: () => {
        setInner(v);
        onChange && onChange(v);
      },
      style: {
        position: 'absolute',
        opacity: 0,
        width: 1,
        height: 1
      }
    }), /*#__PURE__*/React.createElement("span", {
      style: {
        width: 20,
        height: 20,
        boxSizing: 'border-box',
        border: 'var(--border)',
        borderRadius: '50%',
        background: 'var(--paper)',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 'none'
      }
    }, on && /*#__PURE__*/React.createElement("span", {
      style: {
        width: 10,
        height: 10,
        borderRadius: '50%',
        background: 'var(--petrol)'
      }
    })), l);
  }));
}
Object.assign(__ds_scope, { Radio });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Radio.jsx", error: String((e && e.message) || e) }); }

// components/forms/Select.jsx
try { (() => {
function Select({
  label,
  options = [],
  value,
  onChange,
  disabled = false,
  id,
  style
}) {
  const [focused, setFocused] = React.useState(false);
  const iid = id || React.useId();
  return /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: 'var(--font-read)',
      color: 'var(--petrol)',
      ...style
    }
  }, label && /*#__PURE__*/React.createElement("label", {
    htmlFor: iid,
    style: {
      display: 'block',
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: 'var(--track-label)',
      textTransform: 'uppercase',
      marginBottom: 6,
      color: 'var(--petrol)'
    }
  }, label), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("select", {
    id: iid,
    value: value,
    onChange: onChange,
    disabled: disabled,
    onFocus: () => setFocused(true),
    onBlur: () => setFocused(false),
    style: {
      appearance: 'none',
      WebkitAppearance: 'none',
      width: '100%',
      boxSizing: 'border-box',
      fontFamily: 'var(--font-read)',
      fontSize: 15,
      fontWeight: 600,
      color: 'var(--petrol)',
      background: disabled ? 'var(--ivory-deep)' : 'var(--paper)',
      border: 'var(--border)',
      borderRadius: 'var(--radius-sm)',
      padding: '10px 36px 10px 12px',
      cursor: disabled ? 'not-allowed' : 'pointer',
      outline: focused ? '3px solid var(--focus-ring)' : 'none',
      outlineOffset: 2,
      opacity: disabled ? 0.6 : 1
    }
  }, options.map(o => typeof o === 'string' ? /*#__PURE__*/React.createElement("option", {
    key: o,
    value: o
  }, o) : /*#__PURE__*/React.createElement("option", {
    key: o.value,
    value: o.value
  }, o.label))), /*#__PURE__*/React.createElement("span", {
    "aria-hidden": "true",
    style: {
      position: 'absolute',
      right: 12,
      top: '50%',
      transform: 'translateY(-50%) rotate(90deg)',
      fontFamily: 'var(--font-code)',
      fontSize: 13,
      pointerEvents: 'none'
    }
  }, "\u276F")));
}
Object.assign(__ds_scope, { Select });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Select.jsx", error: String((e && e.message) || e) }); }

// components/forms/Switch.jsx
try { (() => {
function Switch({
  checked,
  defaultChecked = false,
  onChange,
  label,
  disabled = false
}) {
  const [inner, setInner] = React.useState(defaultChecked);
  const on = checked ?? inner;
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      fontFamily: 'var(--font-read)',
      fontSize: 15,
      fontWeight: 600,
      color: 'var(--petrol)',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.45 : 1
    }
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    role: "switch",
    checked: on,
    disabled: disabled,
    onChange: e => {
      setInner(e.target.checked);
      onChange && onChange(e.target.checked);
    },
    style: {
      position: 'absolute',
      opacity: 0,
      width: 1,
      height: 1
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'relative',
      width: 40,
      height: 24,
      boxSizing: 'border-box',
      border: 'var(--border)',
      borderRadius: 'var(--radius-sm)',
      background: on ? 'var(--amber)' : 'var(--ivory-deep)',
      transition: 'background var(--dur-ui) var(--ease-ui)',
      flex: 'none'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'absolute',
      top: 2,
      left: on ? 18 : 2,
      width: 16,
      height: 16,
      borderRadius: 3,
      background: 'var(--petrol)',
      transition: 'left var(--dur-ui) var(--ease-ui)'
    }
  })), label);
}
Object.assign(__ds_scope, { Switch });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Switch.jsx", error: String((e && e.message) || e) }); }

// components/navigation/Tabs.jsx
try { (() => {
function Tabs({
  tabs = [],
  value,
  defaultValue,
  onChange
}) {
  const [inner, setInner] = React.useState(defaultValue ?? (tabs[0] && (tabs[0].value ?? tabs[0])));
  const cur = value ?? inner;
  return /*#__PURE__*/React.createElement("div", {
    role: "tablist",
    style: {
      display: 'flex',
      gap: 24,
      borderBottom: '2px solid var(--line)',
      fontFamily: 'var(--font-read)'
    }
  }, tabs.map(t => {
    const v = t.value ?? t;
    const l = t.label ?? t;
    const on = v === cur;
    return /*#__PURE__*/React.createElement("button", {
      key: v,
      role: "tab",
      "aria-selected": on,
      onClick: () => {
        setInner(v);
        onChange && onChange(v);
      },
      style: {
        background: 'none',
        border: 0,
        padding: '10px 0',
        marginBottom: -2,
        fontFamily: 'inherit',
        fontSize: 15,
        fontWeight: on ? 800 : 600,
        color: 'var(--petrol)',
        opacity: on ? 1 : 0.72,
        cursor: 'pointer',
        boxShadow: on ? 'inset 0 -4px 0 var(--amber)' : 'none',
        transition: 'box-shadow var(--dur-ui) var(--ease-ui)'
      }
    }, l);
  }));
}
Object.assign(__ds_scope, { Tabs });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/Tabs.jsx", error: String((e && e.message) || e) }); }

// ui_kits/website/BenchmarkChart.jsx
try { (() => {
function BenchmarkChart({
  cmp,
  selected,
  onSelect
}) {
  const W = 760,
    H = 440,
    m = {
      l: 56,
      r: 24,
      t: 20,
      b: 52
    };
  const [hover, setHover] = React.useState(null);
  const sx = c => m.l + (c - cmp.x.min) / (cmp.x.max - cmp.x.min) * (W - m.l - m.r);
  const sy = s => H - m.b - (s - cmp.y.min) / (cmp.y.max - cmp.y.min) * (H - m.t - m.b);
  const money = v => '$' + (v < 0.1 ? v.toFixed(2) : v.toFixed(2));
  const A = window.FA_ASSETS;
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: `0 0 ${W} ${H}`,
    style: {
      width: '100%',
      display: 'block',
      fontFamily: 'var(--font-read)'
    },
    role: "img",
    "aria-label": cmp.label + ' accuracy against cost'
  }, cmp.y.ticks.map(t => /*#__PURE__*/React.createElement("g", {
    key: 'y' + t
  }, /*#__PURE__*/React.createElement("line", {
    x1: m.l,
    x2: W - m.r,
    y1: sy(t),
    y2: sy(t),
    stroke: "rgba(8,44,52,0.14)",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("text", {
    x: m.l - 10,
    y: sy(t) + 4,
    textAnchor: "end",
    fontSize: "12",
    fontWeight: "700",
    fill: "#082C34",
    style: {
      fontVariantNumeric: 'tabular-nums'
    }
  }, t, "%"))), cmp.x.ticks.map(t => /*#__PURE__*/React.createElement("text", {
    key: 'x' + t,
    x: sx(t),
    y: H - m.b + 22,
    textAnchor: "middle",
    fontSize: "12",
    fontWeight: "700",
    fill: "#082C34",
    style: {
      fontVariantNumeric: 'tabular-nums'
    }
  }, money(t))), /*#__PURE__*/React.createElement("line", {
    x1: m.l,
    x2: W - m.r,
    y1: H - m.b,
    y2: H - m.b,
    stroke: "#082C34",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("line", {
    x1: m.l,
    x2: m.l,
    y1: m.t,
    y2: H - m.b,
    stroke: "#082C34",
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("text", {
    x: W - m.r,
    y: H - 8,
    textAnchor: "end",
    fontSize: "11",
    fontWeight: "800",
    letterSpacing: "1.5",
    fill: "#082C34"
  }, "COST PER TASK \u276F (~ ESTIMATED)"), /*#__PURE__*/React.createElement("text", {
    x: m.l + 8,
    y: m.t + 12,
    fontSize: "11",
    fontWeight: "800",
    letterSpacing: "1.5",
    fill: "#082C34"
  }, "ACCURACY"), cmp.results.map((r, i) => {
    const x = sx(r.cost),
      y = sy(r.score),
      on = selected === i,
      hv = hover === i;
    const lp = r.lp || 'right';
    const tx = lp === 'left' ? x - 20 : lp === 'right' ? x + 20 : x,
      ty = lp === 'bottom' ? y + 30 : lp === 'top' ? y - 36 : y + 4;
    const anchor = lp === 'left' ? 'end' : lp === 'right' ? 'start' : 'middle';
    return /*#__PURE__*/React.createElement("g", {
      key: i,
      style: {
        cursor: 'pointer'
      },
      onClick: () => onSelect(i),
      onMouseEnter: () => setHover(i),
      onMouseLeave: () => setHover(null)
    }, /*#__PURE__*/React.createElement("circle", {
      cx: x,
      cy: y,
      r: "22",
      fill: "transparent"
    }), r.fa ? /*#__PURE__*/React.createElement("g", {
      style: {
        transform: `translate(${x}px, ${y}px) rotate(${hv || on ? 36 : 0}deg) scale(${hv || on ? 1.15 : 1})`,
        transition: 'transform var(--dur-ui) var(--ease-pop)'
      }
    }, /*#__PURE__*/React.createElement("g", {
      transform: "translate(-15 -15) scale(0.3)"
    }, [0, 72, 144, 216, 288].map(a => /*#__PURE__*/React.createElement("rect", {
      key: a,
      x: "39",
      y: "3",
      width: "22",
      height: "50",
      rx: "11",
      transform: 'rotate(' + a + ' 50 50)',
      fill: r.winner ? '#FFB52E' : '#FFF7E8',
      stroke: "#082C34",
      strokeWidth: r.winner ? 0 : 7
    })), !r.winner && [0, 72, 144, 216, 288].map(a => /*#__PURE__*/React.createElement("rect", {
      key: 'f' + a,
      x: "39",
      y: "3",
      width: "22",
      height: "50",
      rx: "11",
      transform: 'rotate(' + a + ' 50 50)',
      fill: "#FFF7E8"
    })))) : /*#__PURE__*/React.createElement("circle", {
      cx: x,
      cy: y,
      r: hv || on ? 9 : 7.5,
      fill: "#FFF7E8",
      stroke: "#082C34",
      strokeWidth: "2.5",
      style: {
        transition: 'r var(--dur-hover) var(--ease-ui)'
      }
    }), on && /*#__PURE__*/React.createElement("circle", {
      cx: x,
      cy: y,
      r: "21",
      fill: "none",
      stroke: "#277C80",
      strokeWidth: "3"
    }), /*#__PURE__*/React.createElement("text", {
      x: tx,
      y: ty,
      textAnchor: anchor,
      fontSize: "13",
      fontWeight: r.fa ? 800 : 600,
      fill: "#082C34"
    }, r.model), /*#__PURE__*/React.createElement("text", {
      x: tx,
      y: ty + 15,
      textAnchor: anchor,
      fontSize: "12",
      fontWeight: "500",
      fill: "rgba(8,44,52,0.72)",
      style: {
        fontVariantNumeric: 'tabular-nums'
      }
    }, r.harness, " \xB7 ", r.score.toFixed(1), "% \xB7 ", r.est ? '~' : '', money(r.cost)));
  }));
}
window.BenchmarkChart = BenchmarkChart;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/website/BenchmarkChart.jsx", error: String((e && e.message) || e) }); }

// ui_kits/website/BenchmarksPage.jsx
try { (() => {
function BenchmarksPage({
  go
}) {
  const {
    Tabs,
    Card,
    Table,
    Tag,
    Link,
    FootnoteMark
  } = window.FastAgentDesignSystem_3898e4;
  const B = window.FA_BENCH,
    A = window.FA_ASSETS;
  const [id, setId] = React.useState('frontier');
  const cmp = B.comparisons.find(c => c.id === id);
  const [sel, setSel] = React.useState(0);
  React.useEffect(() => setSel(0), [id]);
  const r = cmp.results[sel];
  const money = v => '$' + v.toFixed(2);
  const rows = cmp.results.map(x => ({
    h: /*#__PURE__*/React.createElement("span", {
      style: {
        fontWeight: x.fa ? 800 : 500
      }
    }, x.harness),
    m: x.model,
    s: x.score.toFixed(1) + '%',
    c: (x.est ? '~' : '') + money(x.cost),
    t: x.total ? (x.est ? '~' : '') + '$' + x.total.toFixed(2) : '—',
    b: x.badge ? /*#__PURE__*/React.createElement(Tag, {
      variant: "muted"
    }, x.badge) : /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 13,
        color: 'var(--text-muted)'
      }
    }, x.attempts)
  }));
  return /*#__PURE__*/React.createElement("main", null, /*#__PURE__*/React.createElement("section", {
    style: {
      ...faWrap,
      display: 'grid',
      gridTemplateColumns: 'minmax(0,1fr) clamp(140px, 20vw, 230px)',
      gap: 32,
      alignItems: 'end',
      paddingTop: 56
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      paddingBottom: 32
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faLabel,
      marginBottom: 14
    }
  }, B.title, " \xB7 ", B.date), /*#__PURE__*/React.createElement("h1", {
    style: {
      ...faVoice,
      fontSize: 'clamp(48px, 6vw, 72px)',
      lineHeight: 0.98
    }
  }, "Benchmarks"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 18,
      lineHeight: 1.55,
      margin: '18px 0 0',
      maxWidth: 640
    }
  }, "Accuracy against cost per task. Run totals cover 89 tasks with five trials per task unless stated otherwise. ", /*#__PURE__*/React.createElement("strong", null, "~ marks estimated costs."))), /*#__PURE__*/React.createElement("img", {
    src: A + 'illustration/presenter.png',
    alt: "",
    style: {
      width: '100%',
      display: 'block'
    }
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      borderTop: 'var(--border)'
    }
  }), /*#__PURE__*/React.createElement("section", {
    style: {
      ...faWrap,
      paddingTop: 28
    }
  }, /*#__PURE__*/React.createElement(Tabs, {
    tabs: B.comparisons.map(c => ({
      value: c.id,
      label: c.label
    })),
    value: id,
    onChange: setId
  }), /*#__PURE__*/React.createElement("p", {
    style: {
      ...faVoice,
      fontSize: 28,
      lineHeight: 1.2,
      margin: '28px 0 24px',
      maxWidth: 900,
      textWrap: 'pretty'
    }
  }, cmp.claim), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: 24,
      alignItems: 'flex-start'
    }
  }, /*#__PURE__*/React.createElement(Card, {
    style: {
      padding: '16px 16px 8px',
      flex: '999 1 560px',
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement(BenchmarkChart, {
    cmp: cmp,
    selected: sel,
    onSelect: setSel
  })), /*#__PURE__*/React.createElement(Card, {
    variant: r.fa ? 'paper' : 'inset',
    style: {
      flex: '1 1 280px',
      display: 'flex',
      flexDirection: 'column',
      gap: 14
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 8,
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement(Tag, {
    variant: r.fa ? 'accent' : 'outline'
  }, r.harness), r.badge && /*#__PURE__*/React.createElement(Tag, {
    variant: "muted"
  }, r.badge)), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 20,
      fontWeight: 800,
      lineHeight: 1.25
    }
  }, r.model), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: 12
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: faLabel
  }, "Accuracy"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 34,
      fontWeight: 900,
      fontVariantNumeric: 'tabular-nums',
      lineHeight: 1.1
    }
  }, r.score.toFixed(1), "%")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: faLabel
  }, "Cost / task"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 34,
      fontWeight: 900,
      fontVariantNumeric: 'tabular-nums',
      lineHeight: 1.1
    }
  }, r.est ? '~' : '', money(r.cost)))), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      lineHeight: 1.6,
      borderTop: '2px solid var(--line)',
      paddingTop: 12
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "Tokens in / out:"), " ", r.tokens), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "Date:"), " ", r.date), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "Runs:"), " ", r.attempts)), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      lineHeight: 1.55,
      color: 'var(--text-muted)'
    }
  }, r.note), /*#__PURE__*/React.createElement(Link, {
    href: "#",
    onClick: e => e.preventDefault(),
    style: {
      fontSize: 14,
      alignSelf: 'flex-start',
      whiteSpace: 'nowrap'
    }
  }, "View run \u276F"))), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 24
    }
  }, /*#__PURE__*/React.createElement(Table, {
    columns: [{
      key: 'h',
      label: 'Harness'
    }, {
      key: 'm',
      label: 'Model'
    }, {
      key: 's',
      label: 'Accuracy',
      numeric: true
    }, {
      key: 'c',
      label: 'Cost / task',
      numeric: true
    }, {
      key: 't',
      label: 'Run total',
      numeric: true
    }, {
      key: 'b',
      label: 'Status'
    }],
    rows: rows,
    highlight: (row, i) => cmp.results[i].winner,
    footnote: /*#__PURE__*/React.createElement("span", null, "Amber rows are fast-agent\u2019s headline results. Provisional results are pending Terminal-Bench leaderboard review. Costs are estimates at configured rates, not billed spend.")
  }))), /*#__PURE__*/React.createElement("section", {
    style: {
      ...faWrap,
      paddingTop: 72,
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
      gap: 48
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h2", {
    style: {
      ...faVoice,
      fontSize: 34,
      lineHeight: 1.05
    }
  }, "Methodology & disclaimers"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 16,
      lineHeight: 1.6
    }
  }, "Frontier and GPT-5.6 charts use published leaderboard results and provisional submissions. Badges identify provisional results and pricing adjustments. Each ", /*#__PURE__*/React.createElement("em", null, "View run"), " link leads to the source submission or published result."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 10
    }
  }, /*#__PURE__*/React.createElement(Link, {
    href: "#"
  }, "Leaderboard methodology and submission rules"), /*#__PURE__*/React.createElement(Link, {
    href: "#"
  }, "Published leaderboard"))), /*#__PURE__*/React.createElement(Card, {
    variant: "inset"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faLabel,
      marginBottom: 10
    }
  }, "Value (6hr)"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 15,
      lineHeight: 1.6,
      margin: 0
    }
  }, "These experiments use a ", /*#__PURE__*/React.createElement("strong", null, "six-hour agent timeout per trial"), " rather than standard task timeouts. They are not standard leaderboard submissions. Costs use recorded Harbor configured-rate estimates, snapshot 5 September 2026."), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 14
    }
  }, /*#__PURE__*/React.createElement(Link, {
    href: "#"
  }, "Six-hour results, source jobs and cost accounting \u276F")))));
}
window.BenchmarksPage = BenchmarksPage;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/website/BenchmarksPage.jsx", error: String((e && e.message) || e) }); }

// ui_kits/website/DocsPage.jsx
try { (() => {
function DocsPage({
  section,
  toast
}) {
  const {
    Card,
    Link
  } = window.FastAgentDesignSystem_3898e4;
  const A = window.FA_ASSETS;
  const side = ['Getting Started', 'Core Concepts', 'Migrating to 0.10', 'Subagents', 'TUI', 'Codex', 'Migrate Automations', 'Agent Skills', 'Batch Processing', 'GEPA Optimization', 'Structured Outputs', 'Compaction'];
  const [cur, setCur] = React.useState('Getting Started');
  const toc = ['Install or upgrade', 'Run', 'Run a card pack', 'Instruction file', 'Model override'];
  const h2 = {
    fontSize: 24,
    fontWeight: 800,
    margin: '40px 0 12px',
    lineHeight: 1.25
  };
  const p = {
    fontSize: 16,
    lineHeight: 1.6,
    margin: '0 0 14px'
  };
  const code = t => /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-code)',
      fontSize: 14,
      background: 'var(--ivory-deep)',
      padding: '1px 5px',
      borderRadius: 4
    }
  }, t);
  const block = lines => /*#__PURE__*/React.createElement(CopyCode, {
    title: "bash",
    onCopy: () => toast('Copied to clipboard.'),
    lines: lines.map(t => ({
      text: t,
      cmd: true
    })),
    style: {
      margin: '0 0 16px'
    }
  });
  return /*#__PURE__*/React.createElement("main", {
    style: {
      ...faWrap,
      display: 'grid',
      gridTemplateColumns: '220px minmax(0,1fr) 200px',
      gap: 48,
      paddingTop: 40
    }
  }, /*#__PURE__*/React.createElement("nav", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 2,
      position: 'sticky',
      top: 180,
      alignSelf: 'start'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faLabel,
      marginBottom: 10
    }
  }, section), side.map(s => /*#__PURE__*/React.createElement("a", {
    key: s,
    href: "#",
    onClick: e => {
      e.preventDefault();
      setCur(s);
    },
    style: {
      fontSize: 15,
      fontWeight: cur === s ? 800 : 500,
      color: 'var(--petrol)',
      textDecoration: 'none',
      padding: '6px 10px',
      borderRadius: 'var(--radius-sm)',
      background: cur === s ? 'var(--ivory-deep)' : 'transparent'
    }
  }, s))), /*#__PURE__*/React.createElement("article", {
    style: {
      minWidth: 0,
      maxWidth: 720
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      color: 'var(--text-muted)',
      marginBottom: 10
    }
  }, section, " \u276F ", cur), /*#__PURE__*/React.createElement("h1", {
    style: {
      ...faVoice,
      fontSize: 56,
      lineHeight: 1
    }
  }, cur), cur !== 'Getting Started' && /*#__PURE__*/React.createElement("p", {
    style: {
      ...p,
      marginTop: 20,
      color: 'var(--text-muted)'
    }
  }, "Placeholder: this kit only recreates the Getting Started page body. Other docs pages share this layout."), /*#__PURE__*/React.createElement("h2", {
    id: "install",
    style: h2
  }, "Install or upgrade"), block(['uv tool install -U fast-agent-mcp']), /*#__PURE__*/React.createElement("p", {
    style: p
  }, "If you have multiple Python versions installed, pin the one required by fast-agent:"), block(['uv tool install -U fast-agent-mcp --python 3.12']), /*#__PURE__*/React.createElement("h2", {
    style: h2
  }, "Run"), block(['fast-agent go']), /*#__PURE__*/React.createElement("h2", {
    style: h2
  }, "Run a card pack"), block(['fast-agent go --pack analyst --model haiku', 'fast-agent go --pack analyst --pack-registry ./marketplace.json --agent planner --model haiku']), /*#__PURE__*/React.createElement("p", {
    style: p
  }, code('--pack'), " installs the pack into the selected fast-agent home if needed, then launches ", code('go'), " normally. ", code('--model'), " is a fallback for cards without an explicit model setting."), /*#__PURE__*/React.createElement("h2", {
    style: h2
  }, "Instruction file"), block(['fast-agent go -i prompt.md', 'fast-agent go -i https://gist.github.com/....']), /*#__PURE__*/React.createElement("h2", {
    style: h2
  }, "Model override"), block(['fast-agent go --model sonnet']), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      borderTop: 'var(--border)',
      marginTop: 40,
      paddingTop: 18,
      fontSize: 15
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      color: 'var(--text-muted)'
    }
  }, "\u276E Previous: fast-agent"), /*#__PURE__*/React.createElement(Link, {
    href: "#"
  }, "Next: Core Concepts \u276F"))), /*#__PURE__*/React.createElement("aside", {
    style: {
      position: 'sticky',
      top: 180,
      alignSelf: 'start'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faLabel,
      marginBottom: 10
    }
  }, "On this page"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      borderLeft: '2px solid var(--line)',
      paddingLeft: 12,
      fontSize: 14
    }
  }, toc.map((t, i) => /*#__PURE__*/React.createElement("a", {
    key: t,
    href: "#",
    onClick: e => e.preventDefault(),
    style: {
      color: 'var(--petrol)',
      textDecoration: 'none',
      fontWeight: i === 0 ? 800 : 500
    }
  }, t))), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      marginTop: 190
    }
  }, /*#__PURE__*/React.createElement("img", {
    src: A + 'illustration/sitter.png',
    alt: "",
    width: "150",
    style: {
      position: 'absolute',
      left: 16,
      top: -155,
      zIndex: 1,
      display: 'block'
    }
  }), /*#__PURE__*/React.createElement(Card, {
    padding: 16,
    style: {
      paddingTop: 150
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 15,
      fontWeight: 800
    }
  }, "Stuck?"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 14,
      lineHeight: 1.5,
      margin: '4px 0 10px'
    }
  }, "Ask the community on Discord."), /*#__PURE__*/React.createElement(Link, {
    href: "https://discord.gg/xg5cJ7ndN6",
    style: {
      fontSize: 14
    }
  }, "Join Discord \u276F")))));
}
window.DocsPage = DocsPage;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/website/DocsPage.jsx", error: String((e && e.message) || e) }); }

// ui_kits/website/HomePage.jsx
try { (() => {
function HomePage({
  go,
  toast
}) {
  const {
    Button,
    Card,
    Link,
    FootnoteMark,
    Sticker
  } = window.FastAgentDesignSystem_3898e4;
  const A = window.FA_ASSETS;
  const features = [{
    t: 'Extensive model support',
    d: 'Native providers for Anthropic, Google and OpenAI-compatible endpoints. Auto configuration for llama.cpp hosted models.',
    l: [['Model features', 'models']]
  }, {
    t: 'MCP and ACP',
    d: 'Attach MCP servers from config or the command line. Deploy agents over ACP or MCP, with transport diagnostics.',
    l: [['MCP guide', 'mcp'], ['ACP guide', 'acp']]
  }, {
    t: 'Plugin and extend',
    d: 'Write plugins and hooks in plain Python, or use the API directly. Distribute configurations with Card Packs.',
    l: [['Plugin docs', 'agents']]
  }, {
    t: 'Control your context',
    d: 'Template-based system prompts. Install and update Agent Skills, prompt files and agent definitions.',
    l: [['Agent Skills', 'guides'], ['System prompts', 'agents']]
  }];
  return /*#__PURE__*/React.createElement("main", null, /*#__PURE__*/React.createElement("section", {
    style: {
      ...faWrap,
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
      gap: 56,
      alignItems: 'center',
      paddingTop: 72,
      paddingBottom: 72
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faLabel,
      marginBottom: 18
    }
  }, "Coding agent and development toolkit"), /*#__PURE__*/React.createElement("h1", {
    style: {
      ...faVoice,
      fontSize: 'clamp(48px, 6.4vw, 76px)',
      lineHeight: 0.98,
      textWrap: 'balance'
    }
  }, "The harness your model deserves."), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 20,
      lineHeight: 1.5,
      margin: '24px 0 32px',
      maxWidth: 520
    }
  }, "Same model, better results. fast-agent leads on ", /*#__PURE__*/React.createElement("strong", null, "accuracy"), " and ", /*#__PURE__*/React.createElement("strong", null, "cost efficiency"), "."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 16,
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement(Button, {
    size: "lg",
    onClick: () => go('guides')
  }, "Try it now"), /*#__PURE__*/React.createElement(Button, {
    size: "lg",
    variant: "secondary",
    onClick: () => go('guides')
  }, "Migrate your automations"))), /*#__PURE__*/React.createElement(Card, {
    style: {
      position: 'relative',
      padding: 28
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      top: -22,
      right: -10
    }
  }, /*#__PURE__*/React.createElement(Sticker, null, "New results!")), /*#__PURE__*/React.createElement("div", {
    style: faLabel
  }, "Terminal-Bench 2.1 \xB7 September 2026"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 'clamp(56px, 7vw, 88px)',
      fontWeight: 900,
      fontVariantNumeric: 'tabular-nums',
      lineHeight: 1,
      margin: '14px 0 4px',
      letterSpacing: '-0.02em'
    }
  }, "88.3%", /*#__PURE__*/React.createElement(FootnoteMark, {
    n: 1,
    sprite: A + 'sprite.svg',
    size: 18
  })), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 16,
      lineHeight: 1.55,
      margin: '10px 0 18px'
    }
  }, "fast-agent + GPT-5.6 Sol high. 4.5 points above Claude Code + Fable 5, at 61% lower estimated cost per task."), /*#__PURE__*/React.createElement("div", {
    style: {
      borderTop: '2px solid var(--line)',
      paddingTop: 12,
      display: 'flex',
      flexDirection: 'column',
      gap: 12
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 12,
      lineHeight: 1.5,
      color: 'var(--text-muted)'
    }
  }, "1. Provisional, pending leaderboard review. 445 trials, fast-agent 0.9.24. Cost estimated at current Sol pricing."), /*#__PURE__*/React.createElement(Link, {
    href: "#benchmarks",
    onClick: e => {
      e.preventDefault();
      go('benchmarks');
    },
    style: {
      fontSize: 15,
      whiteSpace: 'nowrap'
    }
  }, "All benchmarks \u276F")))), /*#__PURE__*/React.createElement("section", {
    style: {
      background: 'var(--ivory-deep)',
      borderTop: 'var(--border)',
      borderBottom: 'var(--border)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faWrap,
      display: 'grid',
      gridTemplateColumns: 'clamp(140px, 20vw, 240px) minmax(0,1fr)',
      alignItems: 'end',
      gap: 0,
      paddingTop: 56
    }
  }, /*#__PURE__*/React.createElement("img", {
    src: A + 'illustration/pointer.png',
    alt: "",
    style: {
      width: '100%',
      display: 'block',
      position: 'relative',
      zIndex: 1,
      marginRight: -20
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--paper)',
      border: 'var(--border)',
      borderBottom: 0,
      borderRadius: '14px 14px 0 0',
      padding: '32px 36px 40px',
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
      gap: 32,
      alignItems: 'center'
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h2", {
    style: {
      ...faVoice,
      fontSize: 48,
      lineHeight: 1
    }
  }, "Get started"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 17,
      lineHeight: 1.55,
      margin: '12px 0 18px'
    }
  }, "Install and run locally in seconds. One command starts an interactive session with shell tools."), /*#__PURE__*/React.createElement(Link, {
    href: "#guides",
    onClick: e => {
      e.preventDefault();
      go('guides');
    }
  }, "Installation guide")), /*#__PURE__*/React.createElement(CopyCode, {
    onCopy: () => toast('Copied to clipboard.'),
    title: "Terminal",
    lines: [{
      text: 'uvx fast-agent-mcp@latest -x',
      cmd: true
    }, {
      text: 'start an interactive session with shell tools'
    }, {
      text: 'uv tool install -U fast-agent-mcp',
      cmd: true
    }, {
      text: 'install the latest version of fast-agent'
    }, {
      text: 'fast-agent --pack codex',
      cmd: true
    }, {
      text: 'download configuration to use codex'
    }]
  })))), /*#__PURE__*/React.createElement("section", {
    style: {
      ...faWrap,
      paddingTop: 88
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-end',
      gap: 24,
      marginBottom: 36,
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h2", {
    style: {
      ...faVoice,
      fontSize: 48,
      lineHeight: 1
    }
  }, "Simple, extendable agents."), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 18,
      lineHeight: 1.55,
      margin: '14px 0 0',
      maxWidth: 620
    }
  }, "Excellent provider and local model support. Flexible context management. Terminal native and scriptable.")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 12
    }
  }, /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    onClick: () => go('agents')
  }, "Build an agent"), /*#__PURE__*/React.createElement(Button, {
    variant: "secondary",
    onClick: () => go('reference')
  }, "Explore the CLI"))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
      columnGap: 48
    }
  }, features.map(f => /*#__PURE__*/React.createElement("div", {
    key: f.t,
    style: {
      borderTop: 'var(--border)',
      padding: '22px 0 28px',
      display: 'grid',
      gridTemplateColumns: 'minmax(0,1fr) auto',
      gap: 24
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", {
    style: {
      margin: 0,
      fontSize: 20,
      fontWeight: 800
    }
  }, f.t), /*#__PURE__*/React.createElement("p", {
    style: {
      margin: '8px 0 0',
      fontSize: 16,
      lineHeight: 1.6
    }
  }, f.d)), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
      alignItems: 'flex-end',
      paddingTop: 3
    }
  }, f.l.map(([l, r]) => /*#__PURE__*/React.createElement(Link, {
    key: l,
    href: '#' + r,
    onClick: e => {
      e.preventDefault();
      go(r);
    },
    style: {
      fontSize: 15,
      whiteSpace: 'nowrap'
    }
  }, l, " \u276F"))))))));
}
window.HomePage = HomePage;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/website/HomePage.jsx", error: String((e && e.message) || e) }); }

// ui_kits/website/Shell.jsx
try { (() => {
const {
  Wordmark,
  Tabs,
  Link
} = window.FastAgentDesignSystem_3898e4;
const A = '../../assets/';
const faVoice = {
  fontFamily: 'var(--font-voice)',
  fontWeight: 900,
  fontVariationSettings: 'var(--voice-settings)',
  letterSpacing: 'var(--track-voice)',
  margin: 0
};
const faLabel = {
  fontSize: 11,
  fontWeight: 800,
  letterSpacing: 'var(--track-label)',
  textTransform: 'uppercase'
};
const faWrap = {
  maxWidth: 1200,
  margin: '0 auto',
  padding: '0 32px',
  boxSizing: 'border-box'
};
const NAV = [{
  value: 'home',
  label: 'fast-agent'
}, {
  value: 'benchmarks',
  label: 'Benchmarks'
}, {
  value: 'guides',
  label: 'Guides'
}, {
  value: 'agents',
  label: 'Agents'
}, {
  value: 'models',
  label: 'Models'
}, {
  value: 'acp',
  label: 'ACP'
}, {
  value: 'a2a',
  label: 'A2A'
}, {
  value: 'mcp',
  label: 'MCP'
}, {
  value: 'reference',
  label: 'Reference'
}];
function SiteHeader({
  route,
  go
}) {
  const [q, setQ] = React.useState('');
  return /*#__PURE__*/React.createElement("header", {
    style: {
      background: 'var(--ivory)',
      borderBottom: 'var(--border)',
      position: 'sticky',
      top: 0,
      zIndex: 20
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      background: 'var(--petrol)',
      color: 'var(--ivory)',
      fontSize: 14,
      fontWeight: 600,
      textAlign: 'center',
      padding: '8px 16px'
    }
  }, "Join the fast-agent community on ", /*#__PURE__*/React.createElement("a", {
    href: "https://discord.gg/xg5cJ7ndN6",
    style: {
      color: 'var(--ivory)',
      fontWeight: 800,
      textDecoration: 'none',
      boxShadow: 'inset 0 -2px 0 var(--amber)'
    }
  }, "Discord")), /*#__PURE__*/React.createElement("div", {
    style: {
      ...faWrap,
      display: 'flex',
      alignItems: 'center',
      gap: 24,
      height: 64
    }
  }, /*#__PURE__*/React.createElement("a", {
    href: "#home",
    onClick: e => {
      e.preventDefault();
      go('home');
    },
    style: {
      textDecoration: 'none',
      flex: 'none',
      whiteSpace: 'nowrap'
    }
  }, /*#__PURE__*/React.createElement(Wordmark, {
    size: 26,
    assetBase: A
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1
    }
  }), /*#__PURE__*/React.createElement("label", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      border: 'var(--border)',
      borderRadius: 'var(--radius-sm)',
      background: 'var(--paper)',
      padding: '0 10px',
      height: 36,
      width: 240,
      flex: '0 1 240px',
      minWidth: 120
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-code)',
      fontSize: 13
    }
  }, "\u276F"), /*#__PURE__*/React.createElement("input", {
    value: q,
    onChange: e => setQ(e.target.value),
    placeholder: "Search the docs",
    style: {
      border: 0,
      outline: 'none',
      background: 'transparent',
      font: '500 14px var(--font-read)',
      color: 'var(--petrol)',
      flex: 1,
      minWidth: 0
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-code)',
      fontSize: 11,
      border: '2px solid var(--line)',
      borderRadius: 4,
      padding: '0 5px'
    }
  }, "/")), /*#__PURE__*/React.createElement("a", {
    href: "https://github.com/evalstate/fast-agent",
    style: {
      fontFamily: 'var(--font-code)',
      fontSize: 13,
      color: 'var(--petrol)',
      textDecoration: 'none',
      display: 'flex',
      flexDirection: 'column',
      lineHeight: 1.3
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 500,
      whiteSpace: 'nowrap'
    }
  }, "evalstate/fast-agent"), /*#__PURE__*/React.createElement("span", {
    style: {
      color: 'var(--text-muted)',
      fontSize: 12
    }
  }, "GitHub"))), /*#__PURE__*/React.createElement("div", {
    style: {
      ...faWrap,
      overflowX: 'auto'
    }
  }, /*#__PURE__*/React.createElement(Tabs, {
    tabs: NAV,
    value: route,
    onChange: go
  })));
}
function SiteFooter({
  go
}) {
  const col = (title, links) => /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 10
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faLabel,
      color: 'var(--amber)'
    }
  }, title), links.map(([l, r]) => /*#__PURE__*/React.createElement("a", {
    key: l,
    href: '#' + r,
    onClick: e => {
      if (!r.startsWith('http')) {
        e.preventDefault();
        go(r);
      }
    },
    style: {
      color: 'var(--ivory)',
      fontSize: 15,
      fontWeight: 600,
      textDecoration: 'none'
    }
  }, l)));
  return /*#__PURE__*/React.createElement("footer", {
    style: {
      background: 'var(--petrol)',
      color: 'var(--ivory)',
      marginTop: 96
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      ...faWrap,
      padding: '56px 32px 40px',
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
      gap: 32
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
      whiteSpace: 'nowrap'
    }
  }, /*#__PURE__*/React.createElement(Wordmark, {
    size: 30,
    inverse: true,
    assetBase: A
  }), /*#__PURE__*/React.createElement("p", {
    style: {
      margin: 0,
      fontSize: 15,
      lineHeight: 1.6,
      maxWidth: 300,
      opacity: 0.85
    }
  }, "Simple, extendable agents. Code, build and evaluate.")), col('Docs', [['Getting started', 'guides'], ['Agents', 'agents'], ['Models', 'models'], ['Reference', 'reference']]), col('Proof', [['Benchmarks', 'benchmarks'], ['Methodology', 'benchmarks'], ['Articles & blog posts', 'reference']]), col('Community', [['Discord', 'https://discord.gg/xg5cJ7ndN6'], ['X / @llmindsetuk', 'https://x.com/llmindsetuk'], ['GitHub', 'https://github.com/evalstate']])), /*#__PURE__*/React.createElement("div", {
    style: {
      ...faWrap,
      padding: '16px 32px 28px',
      borderTop: '2px solid var(--petrol-2)',
      fontSize: 12,
      display: 'flex',
      justifyContent: 'space-between',
      opacity: 0.8
    }
  }, /*#__PURE__*/React.createElement("span", null, "\xA9 2025-2026 llmindset.co.uk"), /*#__PURE__*/React.createElement("span", {
    style: {
      fontFamily: 'var(--font-code)'
    }
  }, "Made with Zensical")));
}
function CopyCode({
  lines,
  title,
  onCopy,
  style
}) {
  const {
    CodeBlock,
    Button
  } = window.FastAgentDesignSystem_3898e4;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative',
      ...style
    }
  }, /*#__PURE__*/React.createElement(CodeBlock, {
    title: title,
    lines: lines
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      top: title ? 4 : 10,
      right: 10
    }
  }, /*#__PURE__*/React.createElement("button", {
    onClick: onCopy,
    style: {
      background: 'transparent',
      color: 'var(--ivory)',
      border: '2px solid var(--petrol-2)',
      borderRadius: 'var(--radius-sm)',
      font: '700 12px var(--font-read)',
      padding: '3px 9px',
      cursor: 'pointer'
    }
  }, "Copy")));
}
Object.assign(window, {
  SiteHeader,
  SiteFooter,
  CopyCode,
  faVoice,
  faLabel,
  faWrap,
  FA_ASSETS: A
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/website/Shell.jsx", error: String((e && e.message) || e) }); }

// ui_kits/website/data.js
try { (() => {
// Subset of docs/docs/javascripts/homepage-benchmark-data.js (evalstate/fast-agent). Sol costs use current (−20%) pricing, as upstream.
(() => {
  const sol = c => c * 0.8;
  window.FA_BENCH = {
    title: 'Terminal-Bench 2.1',
    date: 'September 2026',
    trials: 445,
    comparisons: [{
      id: 'frontier',
      label: 'Frontier',
      claim: 'fast-agent + GPT-5.6 Sol high scores 88.3%: 4.5 points above Claude Code + Fable 5, at 61% lower estimated API cost with current Sol pricing.',
      x: {
        min: 0.2,
        max: 1.35,
        ticks: [0.25, 0.5, 0.75, 1, 1.25]
      },
      y: {
        min: 76,
        max: 90,
        ticks: [76, 80, 84, 88]
      },
      results: [{
        fa: true,
        winner: true,
        harness: 'fast-agent',
        model: 'GPT-5.6 Sol · high',
        score: 88.31,
        cost: sol(270.13526 / 445),
        est: true,
        tokens: '122.32M / 3.55M',
        date: '2026-07-26',
        attempts: '445 trials · PR #174',
        badge: 'Provisional',
        note: 'fast-agent 0.9.24. Estimated cost applies Sol’s 20% August 2026 price reduction to the original $270.14 source-job total.',
        lp: 'top'
      }, {
        fa: true,
        harness: 'fast-agent',
        model: 'Grok 4.6 · medium',
        score: 390 / 445 * 100,
        cost: 193.4 / 445,
        tokens: '233.46M / 6.25M',
        date: '2026-08-24',
        attempts: '445 trials · PR #221',
        badge: 'Provisional',
        note: 'fast-agent 0.10.10. Cost and tokens aggregated from six linked Harbor source jobs.',
        lp: 'left'
      }, {
        fa: true,
        harness: 'fast-agent',
        model: 'Grok 4.6 · high',
        score: 388 / 445 * 100,
        cost: 238.316166 / 445,
        tokens: '284.77M / 8.38M',
        date: '2026-08-16',
        attempts: '445 trials · PR #212',
        badge: 'Provisional',
        note: 'fast-agent 0.10.9. Cost and tokens aggregated from six linked Harbor source jobs.',
        lp: 'right'
      }, {
        harness: 'Claude Code',
        model: 'Fable 5 · xhigh',
        score: 83.82,
        cost: 1.241955,
        tokens: '194.55M / 9.95M',
        date: '2026-06-07',
        attempts: '445 trials · published',
        note: 'Published Terminal-Bench 2.1 leaderboard row.',
        lp: 'left'
      }, {
        fa: true,
        harness: 'fast-agent',
        model: 'GPT-5.6 Sol · medium',
        score: 365 / 445 * 100,
        cost: sol(211.321445 / 445),
        est: true,
        tokens: '101.95M / 2.53M',
        date: '2026-07-23',
        attempts: '445 trials · PR #170',
        badge: 'Provisional',
        note: 'fast-agent 0.9.21. Estimated cost applies Sol’s 20% price reduction to the original $211.32 total.',
        lp: 'right'
      }, {
        harness: 'Claude Code',
        model: 'Opus 4.8 · high',
        score: 78.88,
        cost: 0.644809,
        tokens: '174.81M / 8.09M',
        date: '2026-07-09',
        attempts: '445 trials · published',
        note: 'Published Terminal-Bench 2.1 leaderboard row.',
        lp: 'right'
      }]
    }, {
      id: 'value',
      label: 'Value (6hr)',
      claim: '82.2–84.5% for ~$30–$78 per run with Luna max, DeepSeek Vision and GLM-5.3-Flash, on a six-hour timeout per trial.',
      x: {
        min: 0.04,
        max: 0.2,
        ticks: [0.05, 0.1, 0.15, 0.2]
      },
      y: {
        min: 81,
        max: 86,
        ticks: [81, 82, 83, 84, 85, 86]
      },
      results: [{
        fa: true,
        harness: 'fast-agent',
        model: 'GLM-5.3-Flash · max',
        score: 376 / 445 * 100,
        cost: 77.53332054 / 445,
        total: 77.53,
        est: true,
        tokens: '1802.45M / 23.97M',
        date: '2026-08-27',
        attempts: '445 trial slots · 6hr timeout',
        badge: '6hr timeout',
        note: 'fast-agent 0.10.11. 376/445 rewarded. Recorded cost coverage 432/445.',
        lp: 'left'
      }, {
        fa: true,
        harness: 'fast-agent',
        model: 'DeepSeek V4 Flash Vision Exp · max',
        score: 370 / 445 * 100,
        cost: 53.68446646 / 445,
        total: 53.68,
        est: true,
        tokens: '1665.55M / 19.34M',
        date: '2026-08-28',
        attempts: '445 trial slots · 6hr timeout',
        badge: '6hr timeout',
        note: 'fast-agent 0.10.13. 370/445 rewarded. Recorded cost coverage 386/445.',
        lp: 'right'
      }, {
        fa: true,
        winner: true,
        harness: 'fast-agent',
        model: 'GPT-5.6 Luna · max',
        score: 366 / 445 * 100,
        cost: 29.54135184 / 445,
        total: 29.54,
        est: true,
        tokens: '647.77M / 8.88M',
        date: '2026-08-28',
        attempts: '445 trial slots · 6hr timeout',
        badge: '6hr timeout',
        note: 'fast-agent 0.10.12. 366/445 rewarded. Recorded cost coverage 424/445.',
        lp: 'right'
      }]
    }, {
      id: 'gpt56',
      label: 'GPT-5.6',
      claim: 'Across three matched GPT-5.6 settings, fast-agent beats OpenAI’s published scores and costs less per task at like-for-like pricing.',
      x: {
        min: 0.2,
        max: 1.0,
        ticks: [0.25, 0.5, 0.75, 1]
      },
      y: {
        min: 74,
        max: 90,
        ticks: [74, 78, 82, 86, 90]
      },
      results: [{
        fa: true,
        winner: true,
        harness: 'fast-agent',
        model: 'GPT-5.6 Sol · high',
        score: 88.31,
        cost: sol(270.13526 / 445),
        est: true,
        tokens: '122.32M / 3.55M',
        date: '2026-07-26',
        attempts: '445 trials · PR #174',
        badge: 'Provisional',
        note: 'fast-agent 0.9.24.',
        lp: 'top'
      }, {
        harness: 'OpenAI',
        model: 'GPT-5.6 Sol · high',
        score: 84.7,
        cost: sol(1.09),
        est: true,
        tokens: '—',
        date: '2026-07-30',
        attempts: 'OpenAI score · repriced cost',
        note: 'Score and original $1.09 per task from OpenAI’s launch chart, repriced −20%.',
        lp: 'left'
      }, {
        fa: true,
        winner: true,
        harness: 'fast-agent',
        model: 'GPT-5.6 Sol · medium',
        score: 365 / 445 * 100,
        cost: sol(211.321445 / 445),
        est: true,
        tokens: '101.95M / 2.53M',
        date: '2026-07-23',
        attempts: '445 trials · PR #170',
        badge: 'Provisional',
        note: 'fast-agent 0.9.21.',
        lp: 'right'
      }, {
        harness: 'OpenAI',
        model: 'GPT-5.6 Sol · medium',
        score: 81.8,
        cost: sol(0.89),
        est: true,
        tokens: '—',
        date: '2026-07-30',
        attempts: 'OpenAI score · repriced cost',
        note: 'Score and original $0.89 per task from OpenAI’s launch chart, repriced −20%.',
        lp: 'left'
      }, {
        fa: true,
        winner: true,
        harness: 'fast-agent',
        model: 'GPT-5.6 Terra · high',
        score: 77.75,
        cost: 133.81 / 445,
        tokens: '171.23M / 3.45M',
        date: '2026-07-18',
        attempts: '445 trials · PR #160',
        badge: 'Provisional',
        note: 'Accuracy, cost and tokens from the submission’s static analysis.',
        lp: 'right'
      }, {
        harness: 'OpenAI',
        model: 'GPT-5.6 Terra · high',
        score: 76.67,
        cost: 0.63,
        tokens: '—',
        date: '2026-07-30',
        attempts: 'OpenAI published',
        note: 'Score and API cost per task from OpenAI’s Terminal-Bench 2.1 cost chart.',
        lp: 'right'
      }]
    }]
  };
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/website/data.js", error: String((e && e.message) || e) }); }

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Link = __ds_scope.Link;

__ds_ns.Burst = __ds_scope.Burst;

__ds_ns.Card = __ds_scope.Card;

__ds_ns.CodeBlock = __ds_scope.CodeBlock;

__ds_ns.FootnoteMark = __ds_scope.FootnoteMark;

__ds_ns.Sticker = __ds_scope.Sticker;

__ds_ns.Table = __ds_scope.Table;

__ds_ns.Tag = __ds_scope.Tag;

__ds_ns.Wordmark = __ds_scope.Wordmark;

__ds_ns.Dialog = __ds_scope.Dialog;

__ds_ns.Loader = __ds_scope.Loader;

__ds_ns.Toast = __ds_scope.Toast;

__ds_ns.Tooltip = __ds_scope.Tooltip;

__ds_ns.Checkbox = __ds_scope.Checkbox;

__ds_ns.Input = __ds_scope.Input;

__ds_ns.Radio = __ds_scope.Radio;

__ds_ns.Select = __ds_scope.Select;

__ds_ns.Switch = __ds_scope.Switch;

__ds_ns.Tabs = __ds_scope.Tabs;

})();
