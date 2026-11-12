Data table for benchmark receipts; always pass a footnote with n, date, model and harness version.

```jsx
<Table caption="Pass rate" columns={[{key:'h',label:'Harness'},{key:'p',label:'Pass',numeric:true}]}
  rows={[{h:'fast-agent 0.3',p:'71.4%'},{h:'Baseline',p:'64.0%'}]} highlight={0}
  footnote="n = 500 tasks, 12 Sep 2026, same model." />
```
