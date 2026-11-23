# LayerDelta v0.1 — แผนพัฒนาและตรวจรับ

> **Status:** M0–M4 complete (internal). The pilot and publication steps are parked under *Release readiness* in the [Roadmap](../roadmap/Roadmap.md).

เริ่มจาก repository ที่มีเอกสารออกแบบ เป้าหมายของ release คือให้ผู้ใช้บน Windows x64 เปิด CNN ตัวอย่าง ตรวจ channel ทดลองแก้หนึ่งจุด ดูว่าภาพใดทายถูกหรือผิดเพิ่ม แล้วส่งสูตรให้ Agent รันซ้ำได้ การวัดที่ถูกต้องและตรวจสอบย้อนหลังได้คือเกณฑ์ส่งมอบ ผลบวกต่อ accuracy เป็นสมมติฐานที่ทดลองได้

อ่านข้อกำหนดเดิมประกอบจาก [PRD และ G1–G10](LayerDelta_PRD_v0.1.md), [Architecture](../architecture/Architecture.md), [HTTP API](../reference/api.md), [คู่มือ Agent](../guides/agent-integration.md) และ [ข้อเสนอโครงการ](LayerDelta_Open_Source_Proposal.md)

## ขอบเขตที่ยืนยันหลังอ่านเอกสาร

| การตัดสินใจ | ผลต่อ implementation |
|---|---|
| MNIST และ CNN สอง Conv blocks หนึ่ง architecture | มี built-in adapter หนึ่งตัว รับ checkpoint อื่นได้เมื่อ state schema ตรงกัน เลื่อนโมเดลทั่วไป การแก้หลายตำแหน่ง การฝึกซ้ำในแอป และ MCP ไปหลัง v0.1 |
| ตรวจหลักฐานก่อนเสนอสูตร | Typed HTTP client ต้องค้นภาพคู่ normal/blurred และขอ preview ราย channel ได้ เพิ่มสองความสามารถนี้จากตาราง tools เดิม ตัวอย่าง headless ต้องจบ workflow ได้โดยใช้ client จริง |
| Windows x64, CPU และ NVIDIA CUDA | ทั้งสอง device เป็นเป้าตรวจรับ ตั้งแต่ preview, baseline, intervention, holdout จนถึงโหลด child กลับมาตรวจผล บันทึก device ที่ใช้งานจริงทุกครั้ง ผู้ใช้เลือกเองว่าจะเปลี่ยนไป CPU หลัง GPU ใช้งานไม่ได้ |
| วัดการแก้ถูกและผลเสีย | เลือกคู่เลขสาธิตจาก baseline บน exploration วัด accuracy ของภาพเบลอที่มี **true label** เป็นคลาสเป้าหมาย เทียบ accuracy บนภาพปกติและแสดงผลข้ามคลาส นับกรณีผิดแล้วเปลี่ยนไปผิดอีกคลาสแยกไว้ |
| คนควบคุมผลยืนยันและ checkpoint | Agent สำรวจภายใน project/session และงบที่กำหนด ผู้ใช้ยืนยัน frozen recipe ก่อนเปิด holdout และยืนยันการ export child จาก weight run ที่สำเร็จ |

## ลำดับงานและจุดส่งต่อ

เส้นทางหลักคือ **fixture ที่ระบุตัวตนได้ → runner ที่ตรวจคำตอบได้ → Desktop และ Agent ที่ใช้ service เดียวกัน → ทดสอบความผิดพลาดและ pilot → ติดตั้งจาก release จริง** งานทดลอง packaging และสำรวจเครื่อง GPU เริ่มคู่กับ fixture ได้ เมื่อ schema ตัวอย่างผ่าน M1 แล้วจึงพัฒนา Desktop และ typed client คู่ขนานกันได้ ทั้งสองฝั่งต้องทดสอบกับ backend จริงก่อนจบ M2

### M0 · ตรึงตัวอย่างและสภาพแวดล้อมตรวจรับ

| งาน | วิธีทำ | หลักฐานก่อนส่งต่อ |
|---|---|---|
| 0.1 สร้าง reference model | กำหนด topology สอง Conv blocks พร้อม bias ตั้งชื่อ layer และ hook หลัง ReLU ระบุ kernel parameter, channel axis, shape, dtype และจุด read-only สร้างสคริปต์เตรียม checkpoint พร้อม seed, training config และที่มา | Adapter specification กับ manifest ที่ชี้ parent Safetensors ด้วย SHA-256 และตัวอย่าง channel แรก/สุดท้ายของแต่ละ target |
| 0.2 สร้าง dataset manifest | แบ่ง train/exploration/holdout ด้วย source ID ก่อนสร้าง normal/blurred กำหนด ID ของแต่ละ variant เก็บ source digest, preprocessing revision, blur parameters และ N รายคลาส | ตรวจ uniqueness, split disjointness และภาพคู่จาก source เดียวกันได้ สร้าง manifest ซ้ำแล้วได้สมาชิกและ digest เดิม |
| 0.3 เลือกโจทย์สาธิต | ใช้ baseline เบื้องต้นบน exploration ตรวจ confusion counts, ความชัดของภาพ และจำนวนตัวอย่างที่ผู้ใช้เปิดตรวจได้ เลือกคู่เลขกับระดับ blur จากผลจริง ตรึงโจทย์และบันทึกเหตุผลก่อนพัฒนา quickstart | ตาราง class/condition, sample IDs ที่ใช้สาธิต, นิยาม primary slice และ normal guardrail ให้ runner ใน M1 ยืนยันผลอ้างอิงอีกครั้ง |
| 0.4 เตรียม Windows CPU/CUDA | ระบุเครื่อง CPU และเครื่อง NVIDIA จริง บันทึกรุ่น GPU, VRAM, driver, OS, Python/PyTorch build ทดลองโหลด checkpoint และทำ forward จาก child process บนทั้งคู่ | ตาราง runtime ที่ทดสอบได้ พร้อมคำสั่งทำซ้ำ หากยังหาเครื่อง GPU ไม่ได้ให้เดินงาน CPU ต่อและคงสถานะ CUDA เป็นรอตรวจรับ |
| 0.5 ทดลอง packaging | ทดลอง desktop shell ที่เปิด React renderer, เปิด/ปิด Python backend, เลือก workspace และพบ runtime จากเครื่องสะอาด ทดสอบเส้นทางติดตั้ง CPU โดยไม่มี NVIDIA และการติดตั้ง CUDA บนเครื่องเป้าหมาย | บันทึกทางเลือก shell และวิธีแจก runtime พร้อมหลักฐาน startup, ขนาดติดตั้งและข้อจำกัด ใช้ผลนี้เลือกวิธีแพ็กก่อนเริ่ม M2 |

M0 จบเมื่อเปิด fixture ได้ ตรวจที่มาและ mapping ได้ และมีชุดข้อมูลสำหรับคำนวณคำตอบ M1 รวมตัวอย่างขนาดเล็กที่คำนวณผลคาดหมายได้เอง เก็บหลักฐานสิทธิ์ใช้และแจก artifacts ตั้งแต่ระยะนี้เพื่อให้เลือกวิธีส่งมอบทัน M4

### M1 · พัฒนา runner จาก CLI จนพิสูจน์ผลได้

1. **Domain และ loader** สร้าง schema สำหรับ adapter, dataset, evaluation, recipe และ run แยก canonical recipe hash ออกจาก execution fingerprint เพื่อให้สูตรเดียวเดินทางข้าม workspace และ device ได้ ก่อนโหลด Safetensors ตรวจขนาด header/file, key set, shape, dtype และจำนวน bytes ตรวจ finite และ strict load บน model instance ใหม่ ใช้ golden recipe ยืนยัน hash และรายงาน dependency mismatch เป็นราย field
2. **Device และ baseline** ให้ CLI รับ device ที่เลือก ตรวจ availability ใน worker ใช้ eval/inference mode บันทึกคำทำนายและ logits ที่ใช้ตรวจข้าม device ต่อ sample ID เก็บ runtime/build, GPU/driver เมื่อเกี่ยวข้อง, dtype, precision/determinism settings, threads และ batch size ใน execution metadata ใช้ข้อมูลที่กระทบ prediction ร่วมกับ checkpoint, adapter, manifest และ evaluator สร้าง fingerprint ทั้ง baseline cache และ preview cache ต้องแยก CPU/CUDA และการตั้งค่าที่กระทบผล
3. **Intervention สองชนิด** แยก process ต่อ run และสร้าง working model จาก parent เดิม สำหรับ activation ให้ hook ที่ output หลัง ReLU เฉพาะ channel ที่เลือก ถอด hook เมื่อจบทุกกรณี สำหรับ weight ให้แก้ kernel slice ของ output channel และคง bias เดิม รองรับ zero และ finite scale โดยใช้ช่วงเสนอ [0, 2] ใน fixture ตรวจ identity ที่ 1, zero เทียบ scale 0, mapping ช่องแรก/ท้าย และกรณี bias ที่ทำให้สองโหมดให้ผลต่างกัน
4. **Paired evaluator** จับคู่ baseline และผลทดลองด้วย sample ID ตรวจ label, split และ condition ให้ตรงกัน ปฏิเสธรายการขาด ซ้ำ หรือเกิน คำนวณ N, accuracy, delta เป็น percentage points, confusion matrix และ counts ของ fixed/regressed/changed-wrong ใช้ true label และ condition นิยาม primary slice ตรึงสมาชิกของ slice ที่เลือกจาก baseline ก่อนทดลอง กลุ่มว่างให้ N = 0 และ metric เป็น null รายงาน overall แบบ micro-average บน sample-condition rows ที่เลือก
5. **CLI PoC และการตรวจข้าม device** รัน baseline → activation → weight → baseline บน CPU และ CUDA แยกกัน เก็บ JSON รายภาพและ hash ของ parent ทดสอบ failure/cancel ระหว่างทดลองแล้วเปิด baseline ใหม่ บน fixture เดียวกันจับคู่ logits CPU/CUDA วัดความต่างและระบุ sample IDs ทุกภาพที่ predicted class ต่างกัน ใช้ผลบนชุด calibration กำหนด tolerance และล็อกก่อนทดสอบ regression เก็บ logits สำหรับการวินิจฉัยไว้ในเครื่อง
6. **Benchmark** บันทึกเวลาทั้ง run แยกจาก inference, batch size, peak RAM/VRAM, ขนาด artifact และ cold/warm cache แยกตามเครื่อง ใช้กำหนดค่า timeout และขนาด preview ใน milestone ถัดไป การตั้งเป้าให้ GPU เร็วกว่า CPU บน CNN ขนาดเล็กอยู่นอกเกณฑ์ตรวจรับ

**จุดผ่าน M1** ผู้พัฒนาเรียก CLI แล้วได้ recipe, baseline ID และผลรายภาพที่ตรวจเทียบกับ fixture ได้ G2–G5 ผ่านบน runtime ที่รองรับ มีรายงานเทียบข้าม device แยกจากการวัดผลของ intervention ถ้า mapping, denominator หรือ isolation ผิด ให้แก้ที่ runner ก่อนเชื่อม UI

### M2 · ต่อ runner เข้ากับ workflow ที่คนและ Agent ใช้จริง

| ลำดับ | สิ่งที่สร้าง | วิธีตรวจผ่าน entry point จริง |
|---|---|---|
| 2.1 Application และ persistence | ใช้ application services ร่วมกันสำหรับ catalog, preview, propose, submit, status, cancel, compare และ export ใช้ SQLite เก็บสถานะและสิทธิ์ ใช้ filesystem เก็บผล immutable เริ่ม supervisor เดียวต่อ workspace และรัน child ทีละงาน บันทึก run, quota charge และ idempotency key ใน transaction เดียว | Submit ซ้ำหลัง response หายได้ run ID เดิมและถูกคิดงบครั้งเดียว งานที่ยังไม่ publish มีสถานะให้ poll ได้ |
| 2.2 HTTP contract | ใช้ FastAPI/Pydantic ตาม endpoint catalog เดิม เพิ่ม capability ของ device และ requested/actual runtime ลงใน schema ที่เกี่ยวข้อง ให้ preview ส่ง sample/source IDs, target, hook, kernel statistics, ค่า mean/max และสัดส่วนศูนย์ของ activation กำหนดชนิดข้อมูล หน่วย การลดมิติ และ schema version ให้ตรงกันใน API กับ client | Contract fixture ที่เปิดภาพ normal/blurred ของ source เดียวกันและ channel เดียวกันได้ทั้ง CPU/CUDA ตรวจค่า summary กับ tensor ที่คำนวณตรง ๆ ตรวจ cache key รวม channel และ device |
| 2.3 Desktop | ต่อหน้า Project/Model Explorer → Experiment Editor → Run Comparison → History & Artifacts → Final Evaluation ให้เลือก runtime ก่อนเริ่ม แสดง device จริง, N, ภาพที่แก้ถูก/เสียเพิ่ม และ before/after รายภาพ รองรับ keyboard, loading, empty และ error states | เปิดจากไอคอน เลือกภาพและ channel ทดลอง activation แล้วสร้าง weight recipe แยก บันทึก run IDs เปิดดูประวัติและ export recipe/report จากผลที่ publish แล้ว |
| 2.4 Typed client และ headless CLI | เปิด backend แบบ loopback ผ่าน CLI ใช้ token จำกัด project/session สร้าง typed tools จาก schema เดียวกับ HTTP เพิ่มการค้น sample และ preview ให้ client ตามงานจริง ตัวอย่างใช้ deterministic stub เรียก inspect → เลือกภาพคู่ → preview → ตรวจงบ → propose → run/status → compare/get_changes ใช้ bounded polling และเก็บ idempotency key ต่อ submission | ตัวอย่างทำงานได้โดยไม่ต้องสมัคร LM provider สมมติฐานอ้าง sample ID, channel และสถิติจริง หยุดเมื่อถึงงบผู้ใช้หรือ quota ของ server ตรวจ payload ที่ส่งไปยัง provider จำลองว่ามีเฉพาะฟิลด์สรุปที่ผู้ใช้อนุญาต ภาพและแผนที่อยู่ในเครื่องโดยค่าเริ่มต้น |
| 2.5 Replay, holdout และ child | Import สูตรใน workspace ใหม่ด้วย digest ยืนยัน frozen recipe ผ่าน human UI ก่อนรัน holdout บันทึก audit และแยก baseline ของ split/device ให้ชัด ใช้ weight patch implementation เดิมสร้าง Safetensors child ระบุ parent/child SHA, recipe hash และ source run ID | เปิด child ผ่าน loader ปกติและรันโดยปิด intervention บน runtime ต้นทาง ตรวจ prediction ต่อภาพตรงกับ source run และ tensor diff จำกัดใน slice ที่อนุมัติ เปิด recipe/report ของทั้งสองโหมดได้ |

**จุดผ่าน M2** Desktop และ headless มี workflow ตั้งแต่เปิด demo ถึง replay ได้ครบ ทดสอบ G1/G6/G7 ขั้น alpha โดยใช้สูตรเดียวกันผ่าน UI และ Agent แล้วเทียบ IDs ของภาพที่เปลี่ยน counts และ metrics บน runtime เดียวกัน ทำซ้ำบน CPU และ CUDA แก้เอกสาร API, Architecture, PRD และคู่มือ Agent ให้ตรงกับ schema ที่ใช้งานจริง โดยเฉพาะ preview และ device selection

### M3 · Beta, recovery และ pilot

| ชุดทดสอบ | กรณีและผลที่ต้องตรวจ |
|---|---|
| Publish, cancel และ restart | เขียน staging บน filesystem เดียวกัน ตรวจ digest ก่อน publish แล้ว commit catalog กับ terminal state จำลอง disk full, worker crash, timeout, crash หลัง publish ก่อน commit และ cancel ชน success ต้องเห็น terminal state เดียว API อ่านเฉพาะผลที่ commit แล้ว restart คง queued jobs, ทำเครื่องหมาย running ที่ขาดตอนเป็น failed/interrupted และกู้ workspace ได้ |
| สิทธิ์และงบ | แยก human session กับ Agent token ตรวจสิทธิ์ก่อนอ่าน object/artifact ทุกครั้ง ทดสอบ project อื่น, scope ผิด, holdout ผ่าน catalog/sample/preview/history/cache/report/URL, Agent export และ CSRF ใน human UI ทดสอบ concurrent submit, key เดิมแต่ payload ต่าง, queue full, quota หมด, retry หลัง failed/cancelled และการหยุดที่งบผู้ใช้แม้ server ยังเหลือ quota |
| ความถูกต้องของข้อมูล | ทดสอบ recipe ผิด schema, hash ไม่ตรง, tensor ผิด shape หรือไม่ finite, sample ขาด/ซ้ำ, source ID ข้าม split, N = 0 และ manifest ผลสูญหาย baseline cache ต้องตัดผลเสียออก คืน error code ที่ UI และ Agent ใช้ตัดสินใจได้ |
| GPU และการกู้ทรัพยากร | เปิดแอปบนเครื่องไม่มี CUDA แล้วใช้ CPU ได้ เมื่อเลือก GPU ที่ใช้ไม่ได้ให้คืนเหตุผลและรอการเลือกใหม่ จำลอง GPU OOM, cancel และ worker exit แล้วรันงานถัดไปได้ ตรวจ parent hash, device จริง และค่า cache หลังสลับเครื่องประมวลผล |
| Clean install และ performance | ทดสอบ Desktop/headless บน Windows CPU และ NVIDIA ตาม support matrix รวมเส้นทางที่เครื่องไม่มี dev environment วัด quickstart, เวลา submit ถึง comparison, peak RAM/VRAM และพื้นที่ดิสก์เทียบกับค่าที่ freeze จาก PoC |
| Pilot | ให้คนทำโจทย์เดียวกันด้วย workbench และสคริปต์อ้างอิง สลับลำดับ วัดเวลาถึงข้อสรุปและการตรวจผลเสีย ให้ Agent รุ่นเดียวกันลองแบบเห็นผลรวมกับแบบเห็นข้อมูลราย channel ภายใต้งบเท่ากัน วัด valid recipes และการทำนายทิศทางผลที่ยังไม่เห็น ทดลอง candidate ที่ freeze บน holdout พร้อม normal guardrail และประวัติเปิด holdout |

**จุดผ่าน M3** มีผล G1–G9 จาก entry points จริง มีรายงาน pilot ที่บอกจำนวนตัวอย่าง ข้อผิดพลาด และข้อจำกัด ผลที่ accuracy เท่าเดิมหรือลดลงเป็นผล pilot ที่รายงานได้

### M4 · Open-source release

1. สร้าง Desktop package และวิธีติดตั้ง headless CLI ตาม packaging spike ตรึง dependency profiles ที่ตรวจแล้ว บันทึก OS, GPU, driver และ runtime ที่ทดสอบจริง พร้อมขั้นตอนตรวจ device เมื่อเริ่มใช้งาน
2. จัดทำ quickstart จากเครื่องสะอาด, ตัวอย่าง Agent แบบตัวเลข, คู่มือเพิ่ม adapter, schema/recipe ตัวอย่าง, release notes และ issue templates ให้ผู้ร่วมพัฒนาแนบ fingerprints และ run IDs ได้
3. เจ้าของโครงการเลือก license ของโค้ด โดย Apache-2.0 เป็นข้อเสนอ ตรวจสิทธิ์ข้อมูลกับ checkpoint แยกกัน เตรียมขั้นตอนให้ผู้ใช้ดาวน์โหลดหรือสร้าง artifact จากแหล่งที่อนุญาตพร้อมตรวจ digest หากแจกไฟล์นั้นโดยตรงไม่ได้
4. ทดสอบ release candidate จาก package ที่จะปล่อยจริงทั้ง CPU และ CUDA เปิด demo, replay ข้าม workspace และ re-import child เก็บผล G1–G10 พร้อมระบุรุ่น package และเครื่องทดสอบก่อนประกาศ v0.1

## ตารางหลักฐานตรวจรับ

| Gate | หลักฐานที่เก็บ | จุดตรวจ |
|---|---|---|
| G1 Quickstart | บันทึก clean install และ comparison จาก Desktop/headless | M2 alpha, M3, M4 package จริง |
| G2 Mapping | target/channel แรกและท้ายตรงกับ tensor/hook ใน fixture ทุก target | M1 |
| G3 Semantics | scale 1, zero, scale 0 และ bias case ทั้งสองโหมด | M1 |
| G4 Isolation | baseline รายภาพกับ parent hash ตรงก่อน/หลัง success, failure, cancel | M1, M3 |
| G5 Metrics | golden counts, confusion, slice membership, delta และ empty slice | M1 |
| G6 UI/API parity | recipe และ runtime เดียวกันได้ counts, metrics, changed IDs ตรงกัน | M2, M3 |
| G7 Export/replay | recipe hash ข้าม workspace คงเดิม โหลด child แล้วได้ผลรายภาพตรงกับ source run และมีหลักฐาน tensor diff | M2, M4 |
| G8 Failure paths | error codes, quota, restart, disk failure และ unavailable device ผ่าน fault suite | M3 |
| G9 Holdout | split-disjoint manifest, audit ของ frozen recipe และผลทดสอบสิทธิ์ Agent ทุกช่องทาง | M3 |
| G10 Release | Windows packages/วิธีติดตั้ง, เอกสาร, provenance, license และรายการ hardware/runtime ที่ผ่าน | M4 |

## งานแรกเมื่อเริ่ม implement

เริ่มที่ 0.1 และ 0.2 เพื่อให้ได้ checkpoint, target mapping และ sample manifest ชุดแรก ควบคู่กับ 0.4–0.5 เพื่อพิสูจน์ทางเปิด worker บน Windows จากนั้นรัน baseline บน exploration เพื่อเลือกโจทย์สาธิตใน 0.3 ผลส่งมอบรอบแรกคือ fixture พร้อม fingerprints และตัวอย่างภาพคู่ที่ runner, UI และ Agent จะใช้อ้างอิงร่วมกัน
