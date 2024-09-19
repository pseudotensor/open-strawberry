import re


def get_turn_title(response_text):
    tag = 'turn_title'
    pattern = fr'<{tag}>(.*?)</{tag}>'
    values = re.findall(pattern, response_text, re.DOTALL)
    values0 = values.copy()
    values = [v.strip() for v in values]
    values = [v for v in values if v]
    if len(values) == 0:
        # then maybe removed too much
        values = values0
    if values:
        turn_title = values[-1]
    else:
        turn_title = response_text[:15] + '...'
    return turn_title


def get_final_answer(response_text, cli_mode=False):
    tag = 'final_answer'
    pattern = fr'<{tag}>(.*?)</{tag}>'
    values = re.findall(pattern, response_text, re.DOTALL)
    values0 = values.copy()
    values = [v.strip() for v in values]
    values = [v for v in values if v]
    if len(values) == 0:
        # then maybe removed too much
        values = values0
    if values:
        if cli_mode:
            response_text = '\n\nFINAL ANSWER:\n\n' + values[-1] + '\n\n'
        else:
            response_text = values[-1]
    else:
        response_text = None
    return response_text


def get_xml_tag_value(response_text, tag, ret_all=True):
    pattern = fr'<{tag}>(.*?)</{tag}>'
    values = re.findall(pattern, response_text, re.DOTALL)
    values0 = values.copy()
    values = [v.strip() for v in values]
    values = [v for v in values if v]
    if len(values) == 0:
        # then maybe removed too much
        values = values0
    if values:
        if ret_all:
            ret = values
        else:
            ret = [values[-1]]
    else:
        ret = []
    return ret
