"""Exercise actual controls in a hidden WebView2 window against its local backend."""

import json
import re
import struct
import time


def exercise(window, device):
    js = window.evaluate_js

    def wait(expression, description):
        end = time.monotonic() + 180
        while time.monotonic() < end:
            alert = js("document.querySelector('[role=alert]')?.innerText")
            if alert:
                raise AssertionError(f"{description}: {alert}")
            value = js(expression)
            if value:
                return value
            time.sleep(0.25)
        text = js("document.body.innerText.slice(0,900)")
        raise AssertionError(f"Timed out at {description}: {text}")

    def button(name):
        find = f"Array.from(document.querySelectorAll('button:not(:disabled)')).find(b=>b.textContent==={json.dumps(name)})"
        wait(f"Boolean({find})", name)
        js(f"{find}.click()")

    def set_value(selector, value):
        quoted = json.dumps(value)
        js(f"""(() => {{
          const el = {selector};
          Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set.call(el, {quoted});
          el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', {{bubbles:true}}));
        }})()""")

    def field(label):
        return f"Array.from(document.querySelectorAll('label')).find(l=>l.firstChild.textContent.trim()==={json.dumps(label)}).querySelector('select,input,textarea')"

    wait("Boolean(document.querySelector('[data-test=preview]:not(:disabled)'))", "open workspace")
    # The checkpoint picker is exposed by the launcher once pywebview is ready (no dialog is opened). A CSP, cookie or
    # bridge break that hides it must fail the smoke run, not surface later as a dead "Import..." button.
    wait("typeof window.pywebview?.api?.choose_checkpoint_path === 'function'",
         "pywebviewready exposes window.pywebview.api.choose_checkpoint_path")
    set_value(field("Requested device"), device)
    button("Preview channel")
    wait("document.querySelectorAll('[data-test=preview-result] canvas').length === 2", "preview tensors")
    evidence = {"device": device, "preview": js("document.querySelector('[data-test=preview-result] p').innerText")}
    if device not in evidence["preview"]:
        raise AssertionError("Preview reported a different device")
    js("document.querySelector('.source-browser summary').click()")
    button("Load first 25 sources")
    choice = field("Available sources")
    wait("Boolean(document.querySelector('.source-browser select:not(:disabled) option[value]:not([disabled])'))",
         "exploration source list")
    browsed = js(f"Array.from(({choice}).options).find(o=>!o.disabled && o.value!==({field('Source image')}).value).value")
    set_value(choice, browsed)
    button("Cancel preview")
    wait("Array.from(document.querySelectorAll('[role=status]')).some(s=>s.innerText==='Preview cancelled.')",
         "cancel changed-source preview")
    button("Preview channel")
    preview = wait("document.querySelector('[data-test=preview-result] p')?.innerText", "preview browsed source")
    if browsed + "/blurred" not in preview or device not in preview:
        raise AssertionError(f"Preview does not match the selected source or device: {preview}")
    evidence["browsed_preview"] = preview
    for mode, target in (("activation", "features.block1.relu.output"), ("weight", "features.block1.conv.kernel")):
        button("Experiment editor")
        set_value(field("Layer and mode"), target)
        set_value("document.querySelector('textarea')", "Test whether zeroing channel 0 changes blurred digit predictions.")
        button("Run intervention")
        wait("Boolean(document.querySelector('.facts dd')) && document.querySelector('h1').innerText==='Run comparison'", mode + " comparison")
        js("""(() => {
          const details = document.querySelector('.facts').closest('details');
          if (details && !details.open) details.querySelector('summary').click();
        })()""")
        evidence[mode] = js("""Object.fromEntries(Array.from(document.querySelectorAll('.facts dt')).map(el=>[el.innerText,el.nextElementSibling.innerText]))""")
        if (not re.fullmatch(r"run_[0-9a-f]{32}", evidence[mode].get("Run", ""))
                or not re.fullmatch(r"[0-9a-f]{64}", evidence[mode].get("Baseline", ""))
                or evidence[mode].get("Actual device", "").split(":")[0] != device):
            raise AssertionError(f"Incomplete or incorrect published provenance: {evidence[mode]}")
        evidence[mode]["primary"] = js("document.querySelector('table tbody tr').innerText")
        button("History & artifacts")
        wait("Boolean(document.querySelector('code'))","frozen recipe")
        evidence[mode]["recipe_hash"] = js("document.querySelector('code').innerText")
    check = "document.querySelector('input[type=checkbox]:not(:disabled)')"
    wait(f"Boolean({check})","weight export confirmation")
    js(f"{check}.click()")
    button("Create child checkpoint")
    wait("Boolean(document.querySelector('a[download]'))", "published checkpoint")
    evidence["child_download"] = js("document.querySelector('a[download]').pathname")
    evidence["tensor_browser"] = exercise_tensors(js, wait, button, set_value)
    evidence["architecture"] = exercise_architecture(js, wait, button, set_value)
    return evidence


def exercise_tensors(js, wait, button, set_value):
    checkpoint = "document.querySelector('[aria-label=Checkpoint]')"
    compare_with = "document.querySelector('[aria-label=\"Compare with\"]')"
    button("Tensors")
    wait("document.querySelector('.tb-context strong')?.textContent === 'Reference fixture'", "reference catalog entry")
    wait("document.querySelector('.tb-summary')?.textContent.includes('6 tensors') && Boolean(document.querySelector('.tb-heatmap'))",
         "read reference tensors")

    button("Import\u2026")
    header = b'{"corpus.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
    header += b" " * (-len(header) % 8)
    data = struct.pack("<Q", len(header)) + header + struct.pack("<4f", 1, -2, 3, 4)
    js(f"""(() => {{
      const input = document.querySelector('[aria-label="Checkpoint file"]');
      const transfer = new DataTransfer();
      transfer.items.add(new File([new Uint8Array({list(data)})], 'smoke-corpus.safetensors'));
      input.files = transfer.files;
      input.dispatchEvent(new Event('change', {{bubbles: true}}));
    }})()""")
    button("Upload")
    wait("document.querySelector('#tb-selector')?.value.startsWith('corpus.weight [') && Boolean(document.querySelector('.tb-heatmap'))",
         "read imported tensor")
    parent = js(f"{checkpoint}.value")
    js("""Array.from(document.querySelectorAll('.tb-segments button')).find(b => b.textContent.startsWith('Values')).click()""")
    wait("Array.from(document.querySelectorAll('[role=gridcell]')).some(el => el.textContent.trim() === '-2')",
         "read imported value")

    js(f"""(() => {{
      const session = JSON.parse(sessionStorage.getItem('layerdelta-session'));
      const base = '/api/v1/projects/' + encodeURIComponent(session.project_id);
      fetch(base + '/checkpoints/' + {json.dumps(parent)} + '/apply', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json', 'X-CSRF-Token': session.csrf, 'Idempotency-Key': crypto.randomUUID()}},
        body: JSON.stringify({{program: [{{op: 'scale', args: {{k: 2}}, select: {{tensor: 'corpus.weight'}}}}],
          hypothesis: 'Double the smoke test checkpoint weights'}})
      }}).then(async r => {{window.smokeApply = {{status: r.status, body: await r.json()}};}});
    }})()""")
    wait("window.smokeApply?.status", "apply edit to imported checkpoint")
    applied = js("window.smokeApply")
    if applied["status"] != 200:
        raise AssertionError(f"apply failed: {applied}")
    child = applied["body"]["result"]["checkpoint"]
    assert child["parent_id"] == parent

    button("Model explorer")
    button("Tensors")
    wait(f"Array.from({checkpoint}?.options ?? []).some(o => o.value === {json.dumps(child['checkpoint_id'])})",
         "load child catalog entry")
    set_value(checkpoint, child["checkpoint_id"])
    wait(f"{compare_with}.value === {json.dumps(parent)} && document.querySelector('.tb-compare-banner')?.textContent.includes('4 values')", "compare child with catalog parent")
    wait("document.querySelectorAll('.tb-heatmap').length === 3 && Boolean(document.querySelector('.tb-delta-key'))",
         "read parent, child and delta values")
    return {"imported": parent, "applied": child["checkpoint_id"], "changed_values": 4}


def exercise_architecture(js, wait, button, set_value):
    picker = "document.querySelector('[aria-label=\"Architecture checkpoint\"]')"
    button("Architecture")
    fixture = wait(
        f"Array.from({picker}?.options ?? []).find(o => o.text === 'Reference fixture')?.value",
        "reference checkpoint in architecture",
    )
    set_value(picker, fixture)
    attached = "document.querySelector('.an-graph-status')?.textContent.includes('Graph attached')"
    wait(f"{attached} || Boolean(document.querySelector('.an-provider select'))", "reference graph options")
    if not js(attached):
        set_value("document.querySelector('.an-provider select')", "reference_cnn")
        button("Attach graph")
    wait(f"{attached} && Boolean(document.querySelector('.an-edge path'))", "reference graph edges")
    button("Group")
    wait("Boolean(document.querySelector('#an-inspector-panel .an-text-button'))", "reference coupling groups")
    js("document.querySelector('#an-inspector-panel .an-text-button').click()")
    wait("Boolean(document.querySelector('.an-index-strip [data-lit=true]'))", "coupling member strip")
    group = js("new URL(location.href).searchParams.get('sel')")
    js("document.querySelector('#an-inspector-panel .an-open').click()")
    wait("Boolean(document.querySelector('#tb-selector') && new URL(location.href).searchParams.get('slice'))",
         "open the selected member in Tensors")
    return {"reference": fixture, "group": group}
