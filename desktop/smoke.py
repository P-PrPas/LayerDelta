"""Exercise actual controls in a hidden WebView2 window against its local backend."""

import json
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
        find = f"Array.from(document.querySelectorAll('button')).find(b=>b.textContent==={json.dumps(name)} && !b.disabled)"
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
    set_value(field("Requested device"), device)
    button("Preview channel")
    wait("document.querySelectorAll('[data-test=preview-result] canvas').length === 2", "preview tensors")
    evidence = {"device": device, "preview": js("document.querySelector('[data-test=preview-result] p').innerText")}
    if device not in evidence["preview"]:
        raise AssertionError("Preview reported a different device")
    for mode, target in (("activation", "features.block1.relu.output"), ("weight", "features.block1.conv.kernel")):
        button("Experiment editor")
        set_value(field("Layer and mode"), target)
        set_value("document.querySelector('textarea')", "Test whether zeroing channel 0 changes blurred digit predictions.")
        button("Run intervention")
        wait("Boolean(document.querySelector('.facts dd')) && document.querySelector('h1').innerText==='Run comparison'", mode + " comparison")
        evidence[mode] = js("""Object.fromEntries(Array.from(document.querySelectorAll('.facts dt')).map(el=>[el.innerText,el.nextElementSibling.innerText]))""")
        evidence[mode]["primary"] = js("document.querySelector('table tbody tr').innerText")
        button("History & artifacts")
        wait("Boolean(document.querySelector('code'))","frozen recipe")
        evidence[mode]["recipe_hash"] = js("document.querySelector('code').innerText")
    check = "document.querySelector('input[type=checkbox]')"
    wait(f"Boolean({check})","weight export confirmation")
    js(f"{check}.click()")
    button("Create child checkpoint")
    wait("Boolean(document.querySelector('a[download]'))", "published checkpoint")
    evidence["child_download"] = js("document.querySelector('a[download]').pathname")
    return evidence
