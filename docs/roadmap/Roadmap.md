# LayerDelta — Roadmap และสถานะโครงการ

เอกสารนี้เป็นแหล่งข้อมูลหลักของ milestone และสถานะปัจจุบัน · อัปเดตล่าสุด 23 พ.ย. 2026

## ทิศทางผลิตภัณฑ์

ให้มนุษย์และ AI agent **เข้าใจ ตรวจ และแก้ weight ของ neural network** ได้สะดวกและตรวจสอบย้อนหลังได้ โดยครอบคลุมหลายตระกูลโมเดล (CNN, U-Net, OCR, detection, segmentation และอื่น ๆ) และไม่ต้องเขียนโค้ดเฉพาะแต่ละโมเดล สถาปัตยกรรมเป้าหมายแบ่งเป็นสามชั้น ([Architecture_Target](../architecture/Architecture_Target.md)):

| ชั้น | ความหมาย | ต้องรันโค้ดของโมเดลไหม |
|---|---|---|
| A · Tensor space | ไฟล์ parameter ทั้งหมด: อ่าน, เทียบ, เลือก, แปลง, ตรวจ, ส่งออก | ไม่ต้อง |
| B · Graph semantics | ความหมายของแต่ละ axis (out/in unit, head, gate) และกลุ่ม axis ที่ผูกกัน (residual, U-Net skip, BN) จาก `torch.export`/ONNX ผ่านตารางกฎราย op ที่เสียบเพิ่มได้ | export เท่านั้น |
| C · Execution | รันโมเดลกับข้อมูลและวัดผลก่อน/หลังด้วย task evaluator (classification, segmentation, detection, OCR) | ต้อง |

**หลักการวาง milestone:** แบ่งตาม capability ไม่ใช้วันที่ ทุก milestone ส่งมอบ capability เดียวกันทั้งทาง **API/agent** และ **Desktop** โดยใช้ payload ชุดเดียวกัน เลข `0.x` จะเพิ่มก็ต่อเมื่อระบบทำสิ่งใหม่ได้จริง

**นโยบาย release:** ห้าม release, publish, tag release หรือประกาศว่า production-ready จนกว่าเจ้าของโครงการจะสั่งอย่างชัดเจน งาน human pilot ย้ายไปอยู่ใน milestone *Release readiness* และไม่เป็น dependency ของงานปัจจุบัน

## สถานะโดยสรุป

| Milestone | สถานะ |
|---|---|
| v0.1 Reference workbench | **Done** (internal; build `0.1.0rc1` ไม่ได้ publish) |
| v0.2 Tensor-space core | **Done** (capability complete, internal; not published) |
| v0.3 Graph semantics & coupling | **In progress** · G1 done, G2 in review, G3 in progress ([board](../Implementation_Plan_v0.3.md)) |
| v0.4 – v0.6 | Not started |
| Release readiness | Parked จนกว่าเจ้าของโครงการจะสั่ง |

รายละเอียดงานและเจ้าของงานอยู่ใน [Implementation_Plan_v0.3](../Implementation_Plan_v0.3.md) (ปัจจุบัน) และ [Implementation_Plan_v0.2](Implementation_Plan_v0.2.md) (เสร็จแล้ว) คำถามที่รอเจ้าของโครงการตัดสินอยู่ใน [QUESTION.md](../../QUESTION.md)

## Milestones

### v0.1 · Reference workbench — Done

- **ทำได้:** ใช้ CNN MNIST ตัวเดียวที่ hardwire ไว้ ทดลอง zero/scale activation หรือ kernel หนึ่งจุด, เทียบผลแบบ paired, replay, export child, ใช้ scoped Agent API, holdout audit ได้ บน Windows CPU/CUDA
- **หลักฐาน:** G1–G9 ผ่าน ([m1](../history/m1-measurements.md)–[m4](../history/m4-validation.md)), ชุดทดสอบ 88 รายการผ่าน
- **ส่งต่อให้รุ่นต่อไป:** store/supervisor, child-process isolation, publish protocol, idempotency/quota/holdout, scoped token, canonical hash, กฎตรวจ safetensors header
- **จะ retire / freeze:** `ReferenceCNN` ในฐานะแกนของผลิตภัณฑ์ (ยังเก็บไว้เป็น test fixture), การจับคู่ relu↔conv ใน `engine.preview`

### v0.2 · Tensor-space core — Done

- **เป้าหมาย:** checkpoint safetensors ใดก็ได้ ตรวจ เทียบ และแก้ได้ โดยไม่ต้องรันโค้ดจากไฟล์
- **คุณค่าต่อผู้ใช้:** เปิด checkpoint ของ U-Net, YOLO, TrOCR, ViT ที่มีอยู่แล้วได้ทันทีในระดับชื่อ tensor และ agent ได้ชุดเครื่องมือ weight-space ที่ deterministic
- **Capability:** streaming reader + tensor table, selector, `describe`/`read_tensor`/`stats`/`diff`/`query`, edit program + recipe v2 + child writer, capability `weights:read`/`weights:write` (รวมการอุดช่องที่ weight รั่วในโค้ด v0.1), trace ขั้นต่ำ, Desktop tensor browser
- **Acceptance:**
  1. corpus ของ toy model ที่นิยามไว้ใน `tests/` ใช้ seeded init และไม่เพิ่ม dependency (ไม่ใช้ timm/ultralytics): ResNet-18-like, U-Net มี concat skip และ ConvTranspose, ViT-tiny, CRNN, backbone แบบ YOLO ทั้งหมดต้องโหลด, describe, diff, apply โปรแกรม 3 op และ replay ได้ child SHA ตรงกัน โดย `src/` ไม่มีโค้ดเฉพาะโมเดล
  2. เมื่อแปลง recipe v1 `zero-block2-channel17` แล้ว child SHA เท่ากับ child ที่ v0.1 export (writer ต้องคง header layout ของ parent)
  3. import checkpoint ที่ไม่ใช่ MNIST ได้ทั้งจาก local path และ streamed upload โดยไม่โหลดทั้งไฟล์เข้าหน่วยความจำ และ catalog ไม่ผูกกับ reference fixture
  4. token ที่ไม่มี `weights:read` ไม่เห็นค่าหรือสถิติ weight **โดยตรง** จาก route ใดเลย (ทดสอบด้วยการไล่ทุก route; activation map + ภาพยังคำนวณย้อนหา weight ได้ ดู R2)
  5. response ของทุก **v2** tool มี `resolved`, `fingerprints`, `side_effects` และ tool ที่ไม่มีผลข้างเคียงต้องได้ `[]` เสมอ ส่วน legacy route ยังคงรูปแบบเดิม
  6. Desktop แสดง tensor table, slice และ diff จาก payload ชุดเดียวกับ v2 API โดยมี smoke test ทั้งบน reference fixture และบน corpus model อย่างน้อยหนึ่งตัว และ UX/UI Designer approve แล้ว
  7. ชุดทดสอบเดิมของ v0.1 ผ่านทั้งหมด
- **Dependencies:** ไม่มีเพิ่มจาก store/API เดิม
- **Risks:** หน่วยความจำกับไฟล์หลาย GB (mmap บน Windows); การปัดเศษของ BF16/F16; tied weights ที่ safetensors แสดงไม่ได้; "channel" ที่ยังไม่มี graph semantics เป็นแค่การเดาจากชื่อ (ต้องแสดง provenance)

### v0.3 · Graph semantics & coupling

- **เป้าหมาย:** ได้ axis role และ coupling group จาก `torch.export` (หลัก) และ ONNX (รอง) ผ่านตารางกฎราย op แบบ plugin พร้อม profile แบบ declarative ส่วน op ที่ไม่รู้จักให้เป็น opaque boundary
- **คุณค่าต่อผู้ใช้:** "channel 17 ของ encoder block 2" หมายถึง unit เดียวกันทั้งใน conv, BN, conv ถัดไป และ skip concat ทำให้แก้แบบมีโครงสร้างได้ถูกต้อง เช่น `permute_units` ที่รักษา function และ prune mask
- **Desktop:** architecture navigator ที่ใช้กับโมเดลใหญ่ได้ (ยุบ/ขยาย module, ค้นหา, highlight coupling, แสดง provenance ของ role) และแก้ระดับ unit ด้วย `plan` → แสดงผลกระทบ → `apply`
- **Acceptance:** บน corpus ของ v0.2: parameter ทุกตัวผูกกับ node หรือถูก flag; `permute_units` บนทุก closed group ให้ max abs diff ≤ 1e-5 (float32, eval mode, seeded random input); zero ผ่าน group ให้ผลเท่ากับ zero ด้วยมือใน module; out axis ของ ConvTranspose = 1; มีรายการ opaque op และปฏิเสธ edit ที่ข้าม boundary; แสดง `coverage` เป็นตัวเลข
- **Design:** [v0.3-graph-semantics.md](../architecture/v0.3-graph-semantics.md) (tasks G1–G8)
- **Dependencies:** selector ของ v0.2 และ module definition ที่รันได้; ต้องตัดสินใจเรื่องรันโค้ดของผู้ใช้ (Q1 ใน [QUESTION.md](../../QUESTION.md))
- **Risks:** `torch.export` ล้มเมื่อเจอ control flow ที่ขึ้นกับข้อมูล (NMS, beam search) ทางแก้คือ export เฉพาะ backbone/neck; ONNX fuse BN เข้า conv จนชื่อไม่ตรงกับ checkpoint; ต้องประเมินความพร้อมของ Torch-Pruning DepGraph

### v0.4 · Execution & task evaluators

- **เป้าหมาย:** รันโมเดลที่แก้แล้วกับข้อมูล และวัดผลแบบ paired ผ่าน evaluator plugin: classification (MNIST ย้ายมาเป็น fixture หนึ่งตัว), segmentation (mIoU/Dice), detection (mAP), OCR (CER/WER) พร้อม activation probe/edit แบบ generic บน graph node
- **คุณค่าต่อผู้ใช้:** ปิดวงจร แก้ → วัดผล ได้ทุกตระกูล
- **Desktop:** run comparison แบบ generic ตามชนิดงาน (slice, guardrail, ตัวอย่างที่ดีขึ้น/แย่ลง)
- **Acceptance:** ผลของ fixture MNIST ไม่เปลี่ยน (regression ของ G2–G9 ผ่านผ่าน plugin); evaluator ทุกตัวมี golden oracle บนชุดข้อมูลเล็ก; holdout gating เหมือนเดิม
- **Dependencies:** v0.3 สำหรับ activation target (การประเมินแบบแก้ weight อย่างเดียวใช้ v0.2 ร่วมกับ module ที่รันได้)
- **Risks:** dataset/preprocessing คือส่วนที่ยังต้องเขียนเฉพาะโมเดลจริง; ขอบเขตความปลอดภัยเมื่อรันโค้ดของผู้ใช้; ความไม่ deterministic ของ GPU (ใช้ class/count เป็นเกณฑ์ replay)

### v0.5 · Agent weight-space programming

- **เป้าหมาย:** agent ใช้ LayerDelta เป็น tooling layer ได้เต็มรูปแบบ: MCP bridge (1 endpoint ≡ 1 client method ≡ 1 MCP tool), trace timeline, งบการอ่านต่อ session, provider adapter ที่มีนโยบายส่งข้อมูลออก (ค่าเริ่มต้นไม่ส่ง tensor/ภาพ)
- **งานวิจัย:** benchmark ว่าการเห็น weight ช่วย agent หรือไม่ โดยใช้ ≥ 3 ตระกูลโมเดลและ pre-register N ไว้ก่อน (สเปกเดิมอยู่ที่ [LayerDelta_Agent_Extension](../history/LayerDelta_Agent_Extension.md)) ผล null ก็รายงานได้
- **Acceptance:** ผลของ MCP ตรงกับ HTTP (parity test); มี contract test ของ tool schema; G12/G13 ฉบับ generalize ผ่าน
- **Risks:** สมมติฐานหลักที่ว่า weight tools ช่วย agent **ยังไม่มีหลักฐาน** (pilot ของ v0.1: −0.20 pp, n = 1)

### v0.6 · Human editing workflow at scale

- **เป้าหมาย:** แก้ weight ได้ครบวงจรโดยไม่ต้องเขียนโค้ด: program/recipe editor, history และ undo (ใช้ parent ที่ immutable), diff view หลาย child, safety preview ก่อน apply, ใช้กับโมเดลระดับพันล้าน parameter ได้ และมี accessibility
- **Acceptance:** กำหนดร่วมกับ UX/UI Designer เมื่อ v0.4 จบ

### Release readiness — Parked (เริ่มเมื่อเจ้าของโครงการสั่งเท่านั้น)

Packaging ของ capability ใหม่, human pilot ตาม [protocol](../history/human-pilot.md), ทดสอบเครื่องที่ไม่มี WebView2, ทดสอบ folder picker ด้วยมือ, release notes และ license/provenance ของ artifact ใหม่

## Risk register

| Risk | ผลกระทบ | Mitigation | Milestone |
|---|---|---|---|
| R1 ยังไม่มีหลักฐานว่า weight tools ช่วย agent | ทิศทางวิจัยอาจได้ผล null | สมมติฐาน (ยังไม่ได้ตรวจเพราะ pilot ถูก park) คือเครื่องมือของ v0.2 มีประโยชน์ต่อคนแม้ผลวิจัยจะเป็น null ส่วน benchmark ใน v0.5 ใช้หลายตระกูลและ pre-register | v0.5 |
| R2 weight รั่วไปยัง token ที่ไม่มีสิทธิ์ (มีในโค้ดแล้ว) | สิทธิ์ใช้ไม่ได้จริง, benchmark ไม่แฟร์ | v0.2-T4a ปิดช่องทางตรง (kernel stats) และ v0.2-T4c ปิด activation map + ภาพต้นฉบับ (ที่เผย weight ทางพีชคณิต) ของ preview สำหรับ token ที่ไม่มี `weights:read`; มีการทดสอบไล่ทุก route; ยังเหลือ: ห้าม agent เข้าถึง workspace filesystem (`report.json`, preview cache) ใน benchmark runner (v0.5) | v0.2 → v0.5 |
| R3 Coverage ของ graph/op ไม่ครบ | ผู้ใช้เข้าใจผิดว่าระบบรู้ความหมายของ axis | provenance บังคับ, ตัวเลข coverage, opaque boundary | v0.3 |
| R4 การรันโค้ดของผู้ใช้เป็น trust boundary ใหม่ | ความปลอดภัย | ต้อง opt-in, รันใน worker ที่แยก, agent สั่งให้รันไม่ได้, รอเจ้าของตัดสิน (Q1) | v0.3–v0.4 |
| R5 Checkpoint ใหญ่บน Windows (mmap, file lock) | ใช้กับโมเดลจริงไม่ได้ | ทดสอบ RSS bound ใน T1, ตรวจ child writer บน Windows | v0.2 |
| R6 มี weight-edit สองชุด (v1 in-process, v2 child writer) | ผลไม่ตรงกัน | ทดสอบความเท่ากันของ SHA ระหว่าง v1 กับ v2 | v0.2–v0.4 |

## การเปลี่ยนแปลงในรอบนี้ (23 พ.ย. 2026)

- เปลี่ยนจาก roadmap แบบ "v0.1 → release" ที่ผูกกับวันที่ มาเป็น ladder ตาม capability v0.2–v0.6
- ย้าย human pilot, การทดสอบ WebView2/folder picker และ tag/release ไปอยู่ใน *Release readiness* (parked)
- นิยาม v0.2 ใหม่: จาก weight preview เฉพาะ Conv + benchmark กลายเป็น Tensor-space core ที่ใช้ได้ทุกโมเดล ส่วนที่ยังเก็บไว้คือการอุดช่อง `weights:read` และ trace ขั้นต่ำ ส่วน `preview_weights` ถูกแทนด้วย `read_tensor` และ benchmark ย้ายไปอยู่ใน v0.5
- การตัดสินใจเรื่องโจทย์ 9→4 / 8→3 ของ benchmark เลื่อนไปอยู่ใน v0.5 เพราะ benchmark จะใช้หลายตระกูลโมเดล
