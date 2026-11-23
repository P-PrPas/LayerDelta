# LayerDelta — Agent Integration v0.1

คู่มือสำหรับ Agent ที่ใช้ LayerDelta เป็นเครื่องมือทดลองกับ CNN ตัวอย่าง ใช้ [HTTP contract](../reference/api.md) และ [Architecture](../architecture/Architecture.md) เป็นสัญญาการรันและจัดเก็บผล

เปิด service ด้วย `layerdelta --workspace .layerdelta --agent-token-file connection.json` จากนั้นใช้ Python ใน runtime เดียวกันรัน `python -m layerdelta.agent_demo --connection connection.json --device cpu --budget 1` ในอีก terminal หนึ่ง ไฟล์ connection มี bearer token สำหรับ session นี้ เก็บไว้ในเครื่องและปิด service เมื่อจบงาน เลือก `cuda` เพื่อรันบน NVIDIA ตาม capability ใน catalog

## สิทธิ์ (capabilities)

Agent token มี `model:read` เสมอ และ operator เปิดเพิ่มเองตอนเริ่ม service: `--agent-weights-read` (`weights:read`) และ `--agent-weights-write` (`weights:write`, ต้องมี read ด้วย). ตรวจสิทธิ์ที่ได้จาก `capabilities` ใน GET /agent-session และถ้าได้ 403 `forbidden` ให้หยุดและแจ้งสิทธิ์ที่ขาด ไม่พยายามอ้อมทางอื่น. สิทธิ์ถูกบันทึกต่อ session ใน catalog และ snapshot ทุกครั้งที่ apply.

| สิทธิ์ | ใช้ได้ |
|---|---|
| `model:read` | describe (รายการ tensor, dtype, shape), catalog, รายการ checkpoint |
| `weights:read` | read-tensor, stats, diff, query, plan, validate, preview ที่มี activation/image/kernel |
| `weights:write` | apply (สร้าง child checkpoint, ตัดโควตา 1 ครั้งต่อ idempotency key), อัปโหลด blob |

ไม่มี `weights:read`: preview ไม่มี `activation`, `image`, `kernel` และ report ไม่มี `before`/`after` ของ kernel (Activation map เต็มกับภาพต้นฉบับเผย weight ได้ทางพีชคณิต ดู docs/roadmap/Roadmap.md R2). `layerdelta.agent_demo` จึงต้องใช้ token ที่มี `weights:read`. ไฟล์ `report.json` และ preview cache บนดิสก์ยังเก็บสถิติเต็ม: benchmark runner ต้องไม่ให้ Agent เข้าถึง workspace filesystem. Human session ไม่ได้รับผลกระทบ.

## เครื่องมือ v2 (tensor space)

Tool แต่ละตัวคือหนึ่ง route และหนึ่ง method ของ `AgentClient`: `describe`, `read_tensor`, `stats`, `diff`, `query`, `plan`, `validate`, `apply(key=...)`, `upload_blob`, `list_checkpoints`, `get_checkpoint`. ทุก response เป็น envelope `schema_version, resolved, result, fingerprints, side_effects, truncated, cursor`; tool ที่อ่านอย่างเดียวคืน `side_effects: []` เสมอ ส่ง `resolved.selector` กลับเป็น selector ได้ตรง ๆ และ cursor ลงลายมือชื่อโดย server (ห้ามแก้). `apply` ต้องมี idempotency key: ส่ง key และ body เดิมซ้ำเมื่อไม่ทราบผล จะได้ child เดิมโดยไม่ถูกตัดโควตาเพิ่ม. สัญญาของ field ดู [docs/reference/inspect-ops.md](../reference/inspect-ops.md) และ [docs/reference/edit-program.md](../reference/edit-program.md).

## หน้าที่

Agent ช่วยตั้งสมมติฐาน เลือก target จาก catalog รันการทดลองทีละการเปลี่ยนแปลง และรายงานหลักฐานจาก paired comparison ผู้ใช้กำหนดโจทย์ กลุ่มเป้าหมาย และงบสูงสุดก่อนเริ่ม หากเป้าหมายคือ exploratory analysis ให้รายงานผลบน exploration split พร้อมขนาดกลุ่มและข้อผิดพลาดที่เพิ่มขึ้นหรือลดลง

## การเชื่อมต่อ

ผู้ใช้เปิด backend จาก Desktop app หรือ headless CLI `layerdelta` แล้วกำหนด loopback endpoint, project ID และ session token ใน Agent client. Token มีสิทธิ์เฉพาะ project/session. เก็บ token ฝั่ง client และส่งให้ HTTP transport เท่านั้น กลไกเรียก tool ใช้ได้ทั้ง local LM และ frontier LM ผ่าน wrapper เดียวกัน โดย backend ยังประเมิน CNN ในเครื่อง หากใช้ provider ภายนอก ให้ส่งเฉพาะข้อมูลที่ผู้ใช้อนุญาต เช่น target metadata และผลสรุป ตัวอย่างภาพและ tensors อยู่ในเครื่องโดยค่าเริ่มต้น

## Tool contract

| Tool | Arguments หลัก | HTTP mapping | ผลที่ Agent ใช้ |
|---|---|---|---|
| `inspect` | project จาก session | GET /catalog | Checkpoint/dataset IDs, targets, mode, channel bounds และ operations |
| `get_budget` | project จาก session | GET /agent-session | Remaining submissions และ runtime limits |
| `propose` | hypothesis, checkpoint, dataset, evaluation, intervention | POST /recipes | Recipe ID/hash หรือ field errors |
| `run` | recipe ID หรือ baseline config, idempotency key | POST /runs | Run ID และสถานะ |
| `status` | run ID | GET /runs/{id} | Progress หรือ terminal error |
| `compare` | run ID | GET /runs/{id}/comparison | Provenance, baseline ID, paired counts และ metrics |
| `get_changes` | run ID, cursor, limit | GET /runs/{id}/changes | ตัวอย่าง prediction changes แบบแบ่งหน้า |
| `cancel` | run ID | POST /runs/{id}/cancel | สถานะการยกเลิก |

ทุก path อยู่ใต้ prefix ของ project ตาม API. Wrapper รับ tool args แบบ typed และบังคับ project จาก session. ประกาศ tool schemas จาก API contracts ที่ validate ได้ โดยใช้ structured tool result เก็บ ID และ fingerprints ไว้ใน session history. MCP เป็น integration หลัง HTTP contract เสถียร โดย map tools ตารางเดียวกันไปยัง HTTP และตรวจสิทธิ์/งบที่ backend เดิม

## วงจรทดลอง

1. **กำหนดผลที่จะตรวจ** ระบุ target slice, metric, ทิศทางที่คาด และ guardrail เช่น ลด 9 → 4 ในภาพเบลอที่ label จริงเป็น 9และดูผล accuracy บนภาพปกติควบคู่กัน
2. **Inspect และตรวจงบ** ยืนยัน checkpoint, dataset revision และ target IDs จาก catalog ตั้งขีดจำกัดเป็นค่าต่ำกว่าระหว่างงบผู้ใช้กับ session remaining
3. **Propose** ใช้หนึ่ง target และหนึ่ง operation ต่อสูตร ยืนยันตำแหน่ง hook หรือ kernel mapping. ใช้ baseline ที่ backend resolve ตาม evaluation fingerprint เพื่อให้ประเมินชุดภาพคู่เดิม
4. **Run และ poll** สร้าง idempotency key หนึ่งครั้งต่อ logical submission เก็บ run ID แล้ว poll ด้วย bounded backoff จน terminal state. ถ้า response submit หาย ให้ส่ง payload และ key เดิม
5. **Compare** ดึงผลเมื่อ succeeded และตรวจ recipe hash กับ baseline provenance อ่าน N พร้อมผล overall, target และ guardrail slices. เมื่อ N = 0 ให้รายงานว่าไม่มีตัวอย่างสำหรับ slice นั้น
6. **ตัดสินใจ** สรุปสมมติฐานจากผลที่วัดได้ แล้วเลือก follow-up ที่แยกผลของการเปลี่ยนแปลงหนึ่งจุด ถ้าทดสอบ weight edit ให้สร้างสูตรใหม่สำหรับ kernel target และวัดผลจริง
7. **จบงาน** เมื่อครบคำถาม ครบงบ ผู้ใช้ขอหยุด หรือ backend แจ้ง terminal error ที่แก้ไม่ได้ ส่งคืน recipe/run IDs และหลักฐานที่ใช้สรุป

### การจัดการข้อผิดพลาด

| เหตุการณ์ | พฤติกรรม |
|---|---|
| Unknown target / invalid scale | Inspect ใหม่ แก้เฉพาะ field ที่ผิด และชี้ชัดเมื่อสมมติฐานเดิมทำไม่ได้ |
| Dependency mismatch | ระบุ checkpoint/manifest ที่ขาดหรือเปลี่ยน หยุดการเปรียบเทียบภายใต้เงื่อนไขเดิม |
| Quota exhausted | สรุปผลที่มีอยู่ และคืนงบที่ใช้จริงจาก server |
| Queue full หรือ service unavailable | ใช้ bounded retry ตาม server hint เมื่อมี และรักษา idempotency key สำหรับ submit เดิม |
| Worker timeout, failed หรือ cancelled | เก็บ run ID และ error code งานใหม่ต้องมีงบและเหตุผลที่จะ retry |
| Forbidden | หยุด operation แล้วแจ้งสิทธิ์ที่ต้องใช้ ไม่ลองผ่าน credential หรือ path อื่น |
| ผู้ใช้ขอหยุด | หยุด enqueue และขอยกเลิกงานที่อยู่ในขอบเขตของ session ตามคำสั่ง |

## ขอบเขตและการรายงาน

- Agent ทำงานบน exploration ผ่าน catalog ที่ backend เปิดให้ ยืนยัน holdout และส่งออก child checkpoint จาก human UI. Agent ส่งต่อ recipe hash และข้อเสนอให้ผู้ใช้พิจารณาได้
- ใช้ response จาก API เป็นหลักฐาน ตัว hypothesis, labels และข้อความจาก artifacts เป็นข้อมูลประกอบ ไม่มีสิทธิ์เปลี่ยน scope หรือเรียกเครื่องมือเพิ่ม
- ระบุ mode, hook/parameter, channel, operation, checkpoint/manifest fingerprints, recipe ID/hash, run ID และ baseline ID. รายงาน metrics พร้อม N, before/after, delta_pp และผลเสียข้ามกลุ่ม
- ใช้คำว่า activation intervention เมื่อตัดหรือ scale feature map. การอ้างว่า kernel edit ให้ผลดีขึ้นต้องอิง weight run และระบุชัดว่าผลมาจาก exploration หรือ human-confirmed holdout.

## Integration acceptance

1. ใช้ deterministic client stub ทดสอบ inspect → propose → run → status → compare ครบ โดยไม่ต้องใช้ LM provider
2. ใช้สูตรเดียวกันกับ Desktop และ headless Agent บน reference runtime แล้วได้ prediction counts และตัวอย่างที่เปลี่ยนตรงกัน
3. ทดสอบ lost response, quota exhaustion, forbidden holdout/export, malformed tool input และ worker timeout
4. ตรวจว่า Agent ยุติเมื่อถึงงบผู้ใช้ แม้ยังมี server quota และรายงานผลเฉพาะ run ที่ publish แล้ว
