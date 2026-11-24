import {useState} from "react";
import type {ErrorDetails} from "./api";

export function ErrorBanner({error, retry}: {error: ErrorDetails; retry?: () => void}) {
  const [copied, setCopied] = useState(false);
  return <div className="ui-banner" data-tone="neg" role="alert">
    <span>{error.message}</span>
    {retry && <button type="button" onClick={retry}>Retry</button>}
    {(error.code || error.requestId) && <small className="ui-error-details">
      {error.code && <code>{error.code}</code>}
      {error.requestId && <>
        <span>Request <code>{error.requestId}</code></span>
        <button type="button" className="ghost" onClick={() => {
          navigator.clipboard.writeText(error.requestId!).then(() => setCopied(true), () => setCopied(false));
        }}>{copied ? "Copied" : "Copy request ID"}</button>
      </>}
    </small>}
  </div>;
}
