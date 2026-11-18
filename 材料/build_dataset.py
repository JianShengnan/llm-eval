# -*- coding: utf-8 -*-
"""
语音导航意图识别 - 大模型专业能力评测数据集构建脚本

输入（同目录）：
  export_11.xlsx                                  标注平台导出数据
  语音导航-FAQ列表-场景导入模板261027.xlsx       意图树 A
  语音导航-储蓄卡业务菜单层级-场景导入模板261027.xlsx  意图树 B

输出（同目录）：
  语音导航意图识别评测数据集.xlsx    数据集 / 意图树 / 构建说明 三个 sheet
  语音导航意图识别评测数据集.jsonl  每行一条样本，含自包含 prompt
  意图树_完整.json                 含节点描述的完整候选集
  意图树_完整.txt                  可直接粘贴进提示词的完整候选集

口径（已与需求方确认）：
  1. 客户当前对话 = export 中说话人为 user 的语句文本；跳过空文本、纯英文、单个符号、单个数字
  2. 历史对话     = 同一会话ID下、轮次序号小于当前行的最近 5 行（含 user 与 agent），
                    带「客户/客服」角色标签，清洗 [p1]/[=hai2]/[n2] 等语音韵律标记
  3. 小模型识别意图 = 机器识别意图
  4. 意图树       = 两个模板文件合并去重后的层级路径列表（不含长描述），每条样本内联
  5. 真实答案     = 人工标注意图名称；剔除名称为空与字面为 NULL 的样本
  6. 模型输出答案 = 空列，评测时由大模型填写
"""
import os
import re
import json
import zipfile
import collections
from xml.etree import ElementTree as ET

BASE = os.path.dirname(os.path.abspath(__file__))
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'

F_EXPORT = 'export_11.xlsx'
F_FAQ = '语音导航-FAQ列表-场景导入模板261027.xlsx'
F_MENU = '语音导航-储蓄卡业务菜单层级-场景导入模板261027.xlsx'
O_XLSX = '语音导航意图识别评测数据集.xlsx'
O_JSONL = '语音导航意图识别评测数据集.jsonl'
O_TREE_JSON = '意图树_完整.json'
O_TREE_TXT = '意图树_完整.txt'

# ----------------------------------------------------------------------------
# 1. 无第三方依赖的 xlsx 读取
# ----------------------------------------------------------------------------
def col_to_idx(ref):
    n = 0
    for ch in re.match(r'([A-Z]+)', ref).group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path):
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
            shared.append(''.join(t.text or '' for t in si.iter(MAIN)))
    root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows = []
    for row in root.findall('.//m:sheetData/m:row', NS):
        cells = {}
        for c in row.findall('m:c', NS):
            t = c.get('t')
            v = c.find('m:v', NS)
            isel = c.find('m:is', NS)
            if t == 'inlineStr' and isel is not None:
                val = ''.join(x.text or '' for x in isel.iter(MAIN))
            elif v is None:
                val = ''
            elif t == 's':
                val = shared[int(v.text)]
            else:
                val = v.text or ''
            cells[col_to_idx(c.get('r'))] = val or ''
        rows.append([cells.get(i, '') for i in range(max(cells) + 1)] if cells else [])
    z.close()
    return rows


def to_table(rows):
    hdr = rows[0]
    w = len(hdr)
    return [dict(zip(hdr, (r + [''] * w)[:w])) for r in rows[1:]]


# ----------------------------------------------------------------------------
# 2. 无第三方依赖的 xlsx 写入
# ----------------------------------------------------------------------------
def col_letter(n):
    s = ''
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def xml_escape(s):
    s = ''.join(ch for ch in str(s) if ch in '\t\n\r' or ord(ch) >= 0x20)
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def cell_xml(ref, value, style):
    v = xml_escape(value)
    if len(v) > 32700:  # Excel 单元格上限 32767 字符
        v = v[:32700] + '…(已截断)'
    return '<c r="%s" s="%d" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>' % (ref, style, v)


def sheet_xml(headers, rows, widths, wrap_cols, row_height=20):
    ncol = len(headers)
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
             '<dimension ref="A1:%s%d"/>' % (col_letter(ncol), len(rows) + 1),
             '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
             'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
             '<sheetFormatPr defaultRowHeight="15"/>',
             '<cols>']
    for i, w in enumerate(widths, start=1):
        parts.append('<col min="%d" max="%d" width="%d" customWidth="1"/>' % (i, i, w))
    parts.append('</cols><sheetData>')
    parts.append('<row r="1" ht="30" customHeight="1">')
    for i, h in enumerate(headers, start=1):
        parts.append(cell_xml('%s1' % col_letter(i), h, 1))
    parts.append('</row>')
    for ri, row in enumerate(rows, start=2):
        parts.append('<row r="%d" ht="%d" customHeight="1">' % (ri, row_height))
        for ci, val in enumerate(row, start=1):
            style = 2 if ci in wrap_cols else 0
            parts.append(cell_xml('%s%d' % (col_letter(ci), ri), val, style))
        parts.append('</row>')
    parts.append('</sheetData>')
    parts.append('<autoFilter ref="A1:%s%d"/>' % (col_letter(ncol), len(rows) + 1))
    parts.append('</worksheet>')
    return ''.join(parts)


STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FF2F5597"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''


def write_xlsx(path, sheets):
    """sheets: list of dict(name, headers, rows, widths, wrap_cols)"""
    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    for i in range(len(sheets)):
        ct.append('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % (i + 1))
    ct.append('</Types>')

    wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i, sh in enumerate(sheets, start=1):
        wb.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (xml_escape(sh['name']), i, i))
        rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (i, i))
    wb.append('</sheets></workbook>')
    rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>' % (len(sheets) + 1))
    rels.append('</Relationships>')

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('[Content_Types].xml', ''.join(ct))
        z.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   '</Relationships>')
        z.writestr('xl/workbook.xml', ''.join(wb))
        z.writestr('xl/_rels/workbook.xml.rels', ''.join(rels))
        z.writestr('xl/styles.xml', STYLES)
        for i, sh in enumerate(sheets, start=1):
            z.writestr('xl/worksheets/sheet%d.xml' % i,
                       sheet_xml(sh['headers'], sh['rows'], sh['widths'], sh.get('wrap_cols', set())))


# ----------------------------------------------------------------------------
# 3. 规则定义
# ----------------------------------------------------------------------------
RE_EN = re.compile(r"^[A-Za-z][A-Za-z0-9_.,'\-?!\s]*$")   # 纯英文（含 sys_err 这类下划线词）
RE_ONE_DIGIT = re.compile(r'^\d$')
RE_ONE_SYMBOL = re.compile(r'^[^\w]$', re.UNICODE)
RE_TAG = re.compile(r'\[[^\]]*\]')                          # [p1] [=hai2] [n2] 等韵律/ASR 标记

GENERIC_INTENTS = {
    '人工服务', '简短词', '集外', '友好问候', '肯定', '否定', '业务咨询', '结束语',
    '答案播报', '投诉建议', '帮助', '抱怨', '老年服务', '业务列表', 'NULL', '无效意图',
    '中银富登', '本地特有服务',
}

# 可选开关：默认 False，严格按需求口径保留历史中的每一行。
# 设为 True 可额外剔除历史对话里残留的英文/单数字/单符号噪声（playover、timeout 等系统信号词）。
FILTER_HISTORY_NOISE = False


def skip_utterance(text):
    s = (text or '').strip()
    if not s:
        return '空文本'
    if RE_EN.fullmatch(s):
        return '英文'
    if RE_ONE_DIGIT.fullmatch(s):
        return '单个数字'
    if RE_ONE_SYMBOL.fullmatch(s):
        return '单个符号'
    return None


def clean_text(text):
    t = RE_TAG.sub('', text or '')
    t = t.replace('[', '').replace(']', '')  # 源文本截断遗留的孤立方括号
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def build_prompt(tree_text, n_nodes, history_text, current, small_intent):
    return (
        '你是银行语音导航系统的意图识别专家。请根据【候选意图树】、【历史对话】、'
        '【客户当前对话】和【小模型识别意图】，从候选意图树中选出唯一一个最匹配的意图节点。\n'
        '输出要求：\n'
        '1. 只能从候选意图树中选择，不得创造树外的意图名称；\n'
        '2. 只输出该节点层级路径的末级名称（即意图名称），不要输出完整路径、编号、引号或任何解释；\n'
        '3. 小模型识别意图仅作参考，若与对话内容冲突，以对话内容为准；\n'
        '4. 若确实无法匹配任何节点，输出「集外」。\n\n'
        '【候选意图树】（共 %d 个节点，每行一个层级路径，"/" 分隔层级）\n%s\n\n'
        '【历史对话】\n%s\n\n'
        '【客户当前对话】\n%s\n\n'
        '【小模型识别意图】\n%s\n\n'
        '【输出】\n'
    ) % (n_nodes, tree_text, history_text or '（无历史对话）', current, small_intent or '（空）')


# ----------------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------------
def main():
    exp = to_table(read_xlsx(os.path.join(BASE, F_EXPORT)))
    faq = to_table(read_xlsx(os.path.join(BASE, F_FAQ)))
    menu = to_table(read_xlsx(os.path.join(BASE, F_MENU)))

    # ---- 4.1 意图树：合并去重后的层级路径 ----
    tree = {}
    for src, tbl in (('FAQ列表', faq), ('储蓄卡业务菜单层级', menu)):
        for r in tbl:
            lv = [r.get(k, '').strip() for k in ('根节点', '子节点', '叶子节点')]
            parts = [x for x in lv if x]
            if not parts:
                continue
            path = '/'.join(parts)
            desc = re.sub(r'\s+', ' ', (r.get('节点描述', '') or '').strip())
            desc_clean = clean_text(r.get('节点描述', '') or '')
            if path not in tree:
                tree[path] = {
                    'path': path, 'level': len(parts),
                    'level1': lv[0], 'level2': lv[1], 'level3': lv[2],
                    'description': desc, 'description_clean': desc_clean, 'sources': [src],
                }
            else:
                if src not in tree[path]['sources']:
                    tree[path]['sources'].append(src)
                if len(desc) > len(tree[path]['description']):
                    tree[path]['description'] = desc
                    tree[path]['description_clean'] = desc_clean
    paths = list(tree.keys())
    tree_text = '\n'.join(paths)
    names = set(p.split('/')[-1] for p in paths)
    n_nodes = len(paths)
    print('[意图树] FAQ %d 行 + 菜单 %d 行 -> 去重路径 %d 个，末级名称 %d 个'
          % (len(faq), len(menu), n_nodes, len(names)))

    # ---- 4.2 会话分组 ----
    sessions = collections.defaultdict(list)
    for excel_row, r in enumerate(exp, start=2):
        r['_excel_row'] = excel_row
        sessions[r['会话ID'].strip()].append(r)
    for sid in sessions:
        sessions[sid].sort(key=lambda x: int(x['轮次序号']) if x['轮次序号'].strip().isdigit() else -1)
    print('[会话] export %d 行，会话 %d 个' % (len(exp), len(sessions)))

    # ---- 4.3 样本抽取 ----
    filter_stat = collections.Counter()
    drop_stat = collections.Counter()
    samples = []
    for sid in sorted(sessions, key=lambda x: int(x) if x.isdigit() else x):
        rows = sessions[sid]
        for r in rows:
            if r['说话人'].strip() != 'user':
                continue
            reason = skip_utterance(r['语句文本'])
            if reason:
                filter_stat[reason] += 1
                continue
            label = r['人工标注意图名称'].strip()
            if not label:
                drop_stat['人工意图名称为空(无效意图)'] += 1
                continue
            if label.upper() == 'NULL':
                drop_stat['字面 NULL'] += 1
                continue

            cur_text = r['语句文本'].strip()
            cur_turn = int(r['轮次序号'])
            prev = [x for x in rows if x['轮次序号'].strip().isdigit()
                    and int(x['轮次序号']) < cur_turn][-5:]
            history = []
            for h in prev:
                txt = clean_text(h['语句文本'])
                if not txt:
                    continue
                if FILTER_HISTORY_NOISE and skip_utterance(txt):
                    continue
                role = '客户' if h['说话人'].strip() == 'user' else '客服'
                history.append({'turn_index': int(h['轮次序号']), 'role': role, 'text': txt})
            history_text = '\n'.join('%s：%s' % (h['role'], h['text']) for h in history)

            human_path = r['人工标注意图'].strip()
            small = r['机器识别意图'].strip()
            stype = '通用交互意图' if label in GENERIC_INTENTS else '业务/FAQ意图'
            samples.append({
                'sample_id': 'S%04d' % (len(samples) + 1),
                'session_id': sid,
                'turn_index': cur_turn,
                'source_row': r['_excel_row'],
                'current_utterance': cur_text,
                'history': history,
                'history_text': history_text,
                'small_model_intent': small,
                'intent_tree': paths,
                'model_answer': '',
                'ground_truth': label,
                'human_intent_path': human_path,
                'is_invalid': r['是否无效意图'].strip(),
                'sample_type': stype,
                'ground_truth_in_tree': label in names,
                'prompt': build_prompt(tree_text, n_nodes, history_text, cur_text, small),
            })
    print('[过滤] ' + ', '.join('%s=%d' % (k, v) for k, v in filter_stat.most_common()))
    print('[剔除] ' + ', '.join('%s=%d' % (k, v) for k, v in drop_stat.most_common()))
    print('[样本] 最终 %d 条' % len(samples))

    # ---- 4.4 输出 JSONL ----
    p_jsonl = os.path.join(BASE, O_JSONL)
    with open(p_jsonl, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # ---- 4.5 输出完整意图树 ----
    p_tree_json = os.path.join(BASE, O_TREE_JSON)
    with open(p_tree_json, 'w', encoding='utf-8') as f:
        json.dump({'node_count': n_nodes, 'source_files': [F_FAQ, F_MENU],
                   'nodes': list(tree.values())}, f, ensure_ascii=False, indent=2)
    p_tree_txt = os.path.join(BASE, O_TREE_TXT)
    with open(p_tree_txt, 'w', encoding='utf-8') as f:
        f.write('# 语音导航意图树（含节点描述）  共 %d 个节点\n' % n_nodes)
        f.write('# 格式：序号. 层级路径  ||  节点描述\n')
        f.write('# 节点描述已清除 [p1]/[=hai2] 等韵律标记；原文见 意图树_完整.json 的 description 字段\n\n')
        for i, p in enumerate(paths, start=1):
            d = tree[p]['description_clean'] or '（无描述）'
            f.write('%d. %s  ||  %s\n' % (i, p, d))

    # ---- 4.6 输出 Excel ----
    headers = ['样本ID', '会话ID', '轮次序号', '客户当前对话', '历史对话',
               '小模型识别意图', '意图树(层级路径)', '评测提示词',
               '模型输出答案(待填)', '真实答案(人工标注意图名称)',
               '人工标注意图(完整路径)', '是否无效意图', '样本类型',
               '答案在候选树中', '源数据行号']
    ds_rows = []
    for s in samples:
        ds_rows.append([
            s['sample_id'], s['session_id'], s['turn_index'], s['current_utterance'],
            s['history_text'], s['small_model_intent'], tree_text, s['prompt'],
            '', s['ground_truth'], s['human_intent_path'], s['is_invalid'],
            s['sample_type'], '是' if s['ground_truth_in_tree'] else '否', s['source_row'],
        ])
    tree_headers = ['序号', '层级路径', '层级数', '一级节点', '二级节点', '三级节点', '节点描述', '来源']
    tree_rows = [[i, p, tree[p]['level'], tree[p]['level1'], tree[p]['level2'],
                  tree[p]['level3'], tree[p]['description'], ' / '.join(tree[p]['sources'])]
                 for i, p in enumerate(paths, start=1)]

    n_hist = sum(len(s['history']) for s in samples)
    n_in_tree = sum(1 for s in samples if s['ground_truth_in_tree'])
    type_stat = collections.Counter(s['sample_type'] for s in samples)
    level_stat = collections.Counter(len(s['human_intent_path'].split('/')) if s['human_intent_path'] else 0 for s in samples)
    small_blank = sum(1 for s in samples if not s['small_model_intent'])
    info = [
        ['一、数据集用途', '用于评测文本大模型在「语音导航意图识别」场景的专业能力：给定对话上下文 + 小模型意图 + 完整候选意图树，要求模型从树中选出唯一意图；人工标注意图名称作为真实答案，用于计算准确率。'],
        ['二、输入字段', '客户当前对话（D列）｜历史对话（E列）｜小模型识别意图（F列）｜意图树层级路径（G列）；H列「评测提示词」为四者拼好的自包含提示词，可直接送大模型。'],
        ['三、输出字段', 'I列「模型输出答案(待填)」为评测时由大模型填写的空列；J列「真实答案(人工标注意图名称)」为 ground truth。评测脚本比对 I 与 J 即可。'],
        ['四、客户当前对话口径', '取 export 中说话人=user 的语句文本原文（未清洗）。跳过：空文本 %d 条、纯英文 %d 条（playover/hangup/timeout/sys_err 等系统信号）、单个数字 %d 条、单个符号 %d 条。' %
         (filter_stat['空文本'], filter_stat['英文'], filter_stat['单个数字'], filter_stat['单个符号'])],
        ['五、历史对话口径', '同一会话ID下、轮次序号小于当前行的最近 5 行（不足 5 行按实际取），含 user 与 agent 两种角色，格式为「客户：…／客服：…」；已清除 [p1]、[=hai2]、[n2]、[h2] 等语音韵律/ASR 标记并压缩空白。清洗后为空的行不展示。共 %d 条样本有历史，平均 %.2f 行。' %
         (sum(1 for s in samples if s['history']), n_hist / max(len(samples), 1))],
        ['六、小模型识别意图', '直接取 export「机器识别意图」列，未做任何改写（含「未识别」等原值）。'],
        ['七、意图树口径', 'FAQ列表 %d 行（单级节点）+ 储蓄卡业务菜单层级 %d 行（最多三级）合并，按层级路径去重后 %d 个节点，末级名称 %d 个。每条样本内联全部路径（不含节点描述）；含描述的完整版见「意图树」sheet 及 意图树_完整.json / .txt。' %
         (len(faq), len(menu), n_nodes, len(names))],
        ['八、样本剔除', '剔除人工标注意图名称为空 %d 条、字面为 NULL %d 条；保留集外、人工服务、友好问候等通用交互意图。' %
         (drop_stat['人工意图名称为空(无效意图)'], drop_stat['字面 NULL'])],
        ['九、最终样本量', '%d 条，来自 %d 个会话；业务/FAQ意图 %d 条，通用交互意图 %d 条；真实答案命中候选树 %d 条（%.2f%%）。' %
         (len(samples), len(set(s['session_id'] for s in samples)), type_stat['业务/FAQ意图'],
          type_stat['通用交互意图'], n_in_tree, 100.0 * n_in_tree / max(len(samples), 1))],
        ['十、答案层级分布', '单级 %d 条，二级 %d 条，三级 %d 条（按人工标注意图路径统计）。' %
         (level_stat[1], level_stat[2], level_stat[3])],
        ['十一、数据校验', '意图树末级名称唯一值 %d 个；真实答案全部能在候选树中找到：%s；小模型意图为空 %d 条；样本ID无重复。' %
         (len(names), '是' if n_in_tree == len(samples) else '否', small_blank)],
        ['十二、JSONL 字段', 'sample_id / session_id / turn_index / source_row / current_utterance / history(数组) / history_text / small_model_intent / intent_tree(路径数组) / prompt(自包含提示词) / model_answer(空) / ground_truth / human_intent_path / is_invalid / sample_type / ground_truth_in_tree。'],
        ['十三、复现方式', '同目录运行 python build_dataset.py 即可重新生成全部文件；所有口径集中在脚本头部注释与常量中。'],
        ['十四、源文件', 'export_11.xlsx（%d 行数据）｜%s｜%s' % (len(exp), F_FAQ, F_MENU)],
        ['十五、历史噪声提示', '历史对话按需求口径保留原行内容（仅清韵律标记），因此仍含 playover / timeout / hangup 等系统信号词及个别英文片段。若不希望这些噪声进入提示词，把 build_dataset.py 中 FILTER_HISTORY_NOISE 改为 True 后重跑即可（默认 False，不改变本次口径）。'],
        ['十六、完整树文件字段', '意图树_完整.json 中 description 为节点描述原文，description_clean 为清除 [p1]/[=hai2] 等韵律标记后的版本；意图树_完整.txt 使用清洗版，可直接贴进提示词。'],
    ]

    write_xlsx(os.path.join(BASE, O_XLSX), [
        {'name': '评测数据集', 'headers': headers, 'rows': ds_rows,
         'widths': [11, 10, 10, 28, 60, 22, 70, 70, 20, 26, 32, 12, 16, 14, 12],
         'wrap_cols': {4, 5, 7, 8}},
        {'name': '意图树', 'headers': tree_headers, 'rows': tree_rows,
         'widths': [8, 58, 9, 22, 26, 28, 90, 26],
         'wrap_cols': {2, 7}},
        {'name': '构建说明', 'headers': ['项目', '说明'], 'rows': info,
         'widths': [24, 130], 'wrap_cols': {2}},
    ])

    print('[输出] ' + O_XLSX)
    print('[输出] ' + O_JSONL)
    print('[输出] ' + O_TREE_JSON)
    print('[输出] ' + O_TREE_TXT)
    print('[校验] 真实答案命中候选树 %d/%d，历史平均 %.2f 行，样本类型 %s'
          % (n_in_tree, len(samples), n_hist / max(len(samples), 1), dict(type_stat)))
    print('[校验] 意图树文本长度 %d 字符，单条 prompt 最大 %d 字符'
          % (len(tree_text), max(len(s['prompt']) for s in samples)))


if __name__ == '__main__':
    main()
