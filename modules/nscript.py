# Libraries
import json
import os, re, textwrap, threading, time, traceback, tiktoken, openai
from pathlib import Path
from colorama import Fore
from dotenv import load_dotenv
from retry import retry
from tqdm import tqdm

# Open AI
load_dotenv()
if os.getenv('api').replace(' ', '') != '':
    openai.base_url = os.getenv('api')
openai.organization = os.getenv('org')
openai.api_key = os.getenv('key')

#Globals
MODEL = os.getenv('model')
TIMEOUT = int(os.getenv('timeout'))
LANGUAGE = os.getenv('language').capitalize()
PROMPT = Path('prompt.txt').read_text(encoding='utf-8')
VOCAB = Path('vocab.txt').read_text(encoding='utf-8')
THREADS = int(os.getenv('threads'))
LOCK = threading.Lock()
WIDTH = int(os.getenv('width'))
LISTWIDTH = int(os.getenv('listWidth'))
NOTEWIDTH = 70
MAXHISTORY = 10
ESTIMATE = ''
TOKENS = [0, 0]
NAMESLIST = []
NAMES = False    # Output a list of all the character names found
BRFLAG = False   # If the game uses <br> instead
FIXTEXTWRAP = True  # Overwrites textwrap
IGNORETLTEXT = False    # Ignores all translated text.
MISMATCH = []   # Lists files that throw a mismatch error (Length of GPT list response is wrong)

#tqdm Globals
BAR_FORMAT='{l_bar}{bar:10}{r_bar}{bar:-10b}'
POSITION = 0
LEAVE = False
PBAR = None
FILENAME = None

# Full Width
ascii_to_wide = dict((i, chr(i + 0xfee0)) for i in range(0x21, 0x7f))
ascii_to_wide.update({0x20: u'\u3000', 0x2D: u'\u2212'})  # space and minus
wide_to_ascii = dict((i, chr(i - 0xfee0)) for i in range(0xff01, 0xff5f))
wide_to_ascii.update({0x3000: u' ', 0x2212: u'-'})        # space and minus

# Pricing - Depends on the model https://openai.com/pricing
# Batch Size - GPT 3.5 Struggles past 15 lines per request. GPT4 struggles past 50 lines per request
# If you are getting a MISMATCH LENGTH error, lower the batch size.
if 'gpt-3.5' in MODEL:
    INPUTAPICOST = .002 
    OUTPUTAPICOST = .002
    BATCHSIZE = 10
elif 'gpt-4' in MODEL:
    INPUTAPICOST = .005
    OUTPUTAPICOST = .015
    BATCHSIZE = 40

def handleOnscripter(filename, estimate):
    global ESTIMATE, FILENAME
    ESTIMATE = estimate
    FILENAME = filename

    if ESTIMATE:
        start = time.time()
        translatedData = openFiles(filename)

        # Print Result
        end = time.time()
        tqdm.write(getResultString(translatedData, end - start, filename))
        with LOCK:
            TOKENS[0] += translatedData[1][0]
            TOKENS[1] += translatedData[1][1]

        # Print Total
        totalString = getResultString(['', TOKENS, None], end - start, 'TOTAL')

        # Print any errors on maps
        if len(MISMATCH) > 0:
            return totalString + Fore.RED + f'\nMismatch Errors: {MISMATCH}' + Fore.RESET
        else:
            return totalString
    
    else:
        try:
            with open('translated/' + filename, 'w', encoding='cp932', errors='ignore') as outFile:
                start = time.time()
                translatedData = openFiles(filename)

                # Print Result
                end = time.time()
                outFile.writelines(translatedData[0])
                tqdm.write(getResultString(translatedData, end - start, filename))
                with LOCK:
                    TOKENS[0] += translatedData[1][0]
                    TOKENS[1] += translatedData[1][1]
        except Exception as e:
            traceback.print_exc()
            return 'Fail'

    return getResultString(['', TOKENS, None], end - start, 'TOTAL')

def getResultString(translatedData, translationTime, filename):
    # File Print String
    totalTokenstring =\
        Fore.YELLOW +\
        '[Input: ' + str(translatedData[1][0]) + ']'\
        '[Output: ' + str(translatedData[1][1]) + ']'\
        '[Cost: ${:,.4f}'.format((translatedData[1][0] * .001 * INPUTAPICOST) +\
        (translatedData[1][1] * .001 * OUTPUTAPICOST)) + ']'
    timeString = Fore.BLUE + '[' + str(round(translationTime, 1)) + 's]'

    if translatedData[2] == None:
        # Success
        return filename + ': ' + totalTokenstring + timeString + Fore.GREEN + u' \u2713 ' + Fore.RESET

    else:
        # Fail
        try:
            raise translatedData[2]
        except Exception as e:
            traceback.print_exc()
            errorString = str(e) + Fore.RED
            return filename + ': ' + totalTokenstring + timeString + Fore.RED + u' \u2717 ' +\
                errorString + Fore.RESET

def openFiles(filename):
    with open('files/' + filename, 'r', encoding='cp932') as readFile:
        translatedData = parseOnscripter(readFile, filename)

        # Delete lines marked for deletion
        finalData = []
        for line in translatedData[0]:
            if line != '\\d\n':
                finalData.append(line)
        translatedData[0] = finalData
    
    return translatedData

def parseOnscripter(readFile, filename):
    totalTokens = [0,0]

    # Read File into data
    data = readFile.readlines()

    # Create Progress Bar
    with tqdm(bar_format=BAR_FORMAT, position=POSITION, leave=LEAVE) as pbar:
        pbar.desc=filename

        try:
            result = translateOnscripter(data, pbar, filename, [])
            totalTokens[0] += result[0]
            totalTokens[1] += result[1]
        except Exception as e:
            traceback.print_exc()
            return [data, totalTokens, e]
    return [data, totalTokens, None]

def translateOnscripter(data, pbar, filename, translatedList):
    stringList = []
    currentGroup = []
    tokens = [0,0]
    speaker = ''
    voice = False
    global LOCK, ESTIMATE, PBAR
    PBAR = pbar
    i = 0

    # Dialogue
    while i < len(data):
        # Lines
        regex = r'^([\u3000「（【[][^\n]+)'
        match = re.search(regex, data[i])
        if match != None and match.group(1) != '':
            originalString = match.group(1)
            # Pass 1
            if translatedList == []:
                # Grab Consecutive Strings
                jaString = match.group(1)
                while len(data) > i+1 and re.match(regex, data[i+1]):
                    data[i] = ''
                    i += 1
                    jaString = f'{jaString} {data[i]}'

                # Convert from Wide
                jaString = jaString.translate(wide_to_ascii)
                    
                # Remove any textwrap and \u3000 and \
                jaString = jaString.replace('\n', '')
                jaString = jaString.replace('\u3000', '')
                jaString = jaString.replace('\\', '')
                jaString = jaString.replace('　＞', ')')
                jaString = jaString.replace('＜　', '(')

                # Remove Furigana
                furiMatch = re.findall(r'({(.+?)\/(.+?)})', jaString)
                if furiMatch:
                    for match in furiMatch:
                        jaString = jaString.replace(match[0], match[2])

                # Add String
                stringList.append(jaString.strip())
            
            # Pass 2
            else:
                # Get Text
                if translatedList:
                    # Grab and Pop
                    translatedText = translatedList[0]
                    translatedList.pop(0)

                    # Set to None if empty list
                    if len(translatedList) <= 0:
                        translatedList = None

                    # Textwrap & Other Text
                    translatedText = textwrap.fill(translatedText, width=WIDTH)
                    translatedText = translatedText.replace('\n', '\n\u3000')

                    # Split the string into lines
                    lines = translatedText.split('\n')
                    
                    # Add a backslash after every 3rd line
                    j = 0
                    while j < len(lines):
                        if j == 4:
                            lines[j-1] = f'{lines[j-1]}\\'
                            lines[j] = f'\n{lines[j]}'
                        j += 1
                    
                    # Join the lines back into a single string
                    translatedText = '\n'.join(lines)

                    # Remove Double Spaces
                    translatedText = translatedText.replace('  ', ' ')

                    # Convert to Wide
                    translatedText = translatedText.translate(ascii_to_wide)

                    # Fix Formatting
                    translatedText = fixText(translatedText)

                    # Set Data
                    data[i] = data[i].replace(originalString, f'{translatedText}')
            i += 1

        # Choices
        elif 'csel' in data[i] and translatedList != []:
            choiceList = []
            jaString = data[i]

            choiceList = re.findall(r'\"(.*?)\"', jaString)
            if len(choiceList) > 0:
                # Translate
                response = translateGPT(choiceList, 'This will be a dialogue option', True)
                translatedTextList = response[0]
                tokens[0] += response[1][0]
                tokens[1] += response[1][1]

                # Set Data
                for j in range(len(translatedTextList)):
                    # Convert to Wide
                    translatedText = translatedTextList[j].translate(ascii_to_wide)

                    # Set
                    data[i] = data[i].replace(choiceList[j], translatedText)
            i += 1

        # Nothing relevant. Skip Line.
        else:
            i += 1

    # EOF
    if len(stringList) > 0:
        # Set Progress
        pbar.total = len(stringList)
        pbar.refresh()
        
        # Translate
        response = translateGPT(stringList, '', True)
        tokens[0] += response[1][0]
        tokens[1] += response[1][1]
        translatedList = response[0]

        # Set Strings
        if len(stringList) == len(translatedList):
            translateOnscripter(data, pbar, filename, translatedList)

        # Mismatch
        else:
            with LOCK:
                if filename not in MISMATCH:
                    MISMATCH.append(filename)
    return tokens

def fixText(translatedText):
    # Add Break
    translatedText = translatedText.replace('\"', '\'')
    translatedText = f'\u3000{translatedText}\\'

    # Unconvert Codes
    matchList = re.findall(r'([＄].+?)[^\w]', translatedText)
    if matchList:
        for match in matchList:
            translatedText = translatedText.replace(match, match.translate(wide_to_ascii))

    # Unconvert Color Codes
    matchList = re.findall(r'([＃][\w\d]{6})', translatedText)
    if matchList:
        for match in matchList:
            translatedText = translatedText.replace(match, match.translate(wide_to_ascii))

    # Unconvert Variables
    matchList = re.findall(r'([％]\w.+?)[^\w＿]', translatedText)
    if matchList:
        for match in matchList:
            translatedText = translatedText.replace(match, match.translate(wide_to_ascii))

    # Unconvert Backslashes
    matchList = re.findall(r'＼', translatedText)
    if matchList:
        for match in matchList:
            translatedText = translatedText.replace(match, match.translate(wide_to_ascii))

    return translatedText

# Save some money and enter the character before translation
def getSpeaker(speaker):
    match speaker:
        case 'ファイン':
            return ['Fine', [0,0]]
        case '':
            return ['', [0,0]]
        case _:
            # Store Speaker
            if speaker not in str(NAMESLIST):
                response = translateGPT(speaker, 'Reply with the '+ LANGUAGE +' translation of the NPC name.', False)
                response[0] = response[0].title()
                response[0] = response[0].replace("'S", "'s")

                # Retry if name doesn't translate for some reason
                if re.search(r'([a-zA-Z？?])', response[0]) == None:
                    response = translateGPT(speaker, 'Reply with the '+ LANGUAGE +' translation of the NPC name.', False)
                    response[0] = response[0].title()
                    response[0] = response[0].replace("'S", "'s")

                speakerList = [speaker, response[0]]
                NAMESLIST.append(speakerList)
                return response
            # Find Speaker
            else:
                for i in range(len(NAMESLIST)):
                    if speaker == NAMESLIST[i][0]:
                        return [NAMESLIST[i][1],[0,0]]
                               
    return [speaker,[0,0]]

def subVars(jaString):
    jaString = jaString.replace('\u3000', ' ')

    # Nested
    count = 0
    nestedList = re.findall(r'[\\]+[\w]+\[[\\]+[\w]+\[[0-9]+\]\]', jaString)
    nestedList = set(nestedList)
    if len(nestedList) != 0:
        for icon in nestedList:
            jaString = jaString.replace(icon, '[Nested_' + str(count) + ']')
            count += 1

    # Icons
    count = 0
    iconList = re.findall(r'[\\]+[iIkKwWaA]+\[[0-9]+\]', jaString)
    iconList = set(iconList)
    if len(iconList) != 0:
        for icon in iconList:
            jaString = jaString.replace(icon, '[Ascii_' + str(count) + ']')
            count += 1

    # Colors
    count = 0
    colorList = re.findall(r'([\\]+c\[\d+\][\\]+c|[\\]+c\[\d+\])', jaString)
    colorList = set(colorList)
    if len(colorList) != 0:
        for color in colorList:
            jaString = jaString.replace(color, '[Color_' + str(count) + ']')
            count += 1

    # Names
    count = 0
    nameList = re.findall(r'[\\]+[nN]\[.+?\]+', jaString)
    nameList = set(nameList)
    if len(nameList) != 0:
        for name in nameList:
            jaString = jaString.replace(name, '[Noun_' + str(count) + ']')
            count += 1

    # Variables
    count = 0
    varList = re.findall(r'[\\]+[vV]\[[0-9]+\]', jaString)
    varList = set(varList)
    if len(varList) != 0:
        for var in varList:
            jaString = jaString.replace(var, '[Var_' + str(count) + ']')
            count += 1

    # Formatting
    count = 0
    formatList = re.findall(r'[\\]+[\w]+\[[a-zA-Z0-9\\\[\]\_,\s-]+\]', jaString)
    formatList = set(formatList)
    if len(formatList) != 0:
        for var in formatList:
            jaString = jaString.replace(var, '[FCode_' + str(count) + ']')
            count += 1

    # Put all lists in list and return
    allList = [nestedList, iconList, colorList, nameList, varList, formatList]
    return [jaString, allList]

def resubVars(translatedText, allList):
    # Fix Spacing and ChatGPT Nonsense
    matchList = re.findall(r'\[\s?.+?\s?\]', translatedText)
    if len(matchList) > 0:
        for match in matchList:
            text = match.strip()
            translatedText = translatedText.replace(match, text)

    # Nested
    count = 0
    if len(allList[0]) != 0:
        for var in allList[0]:
            translatedText = translatedText.replace('[Nested_' + str(count) + ']', var)
            count += 1

    # Icons
    count = 0
    if len(allList[1]) != 0:
        for var in allList[1]:
            translatedText = translatedText.replace('[Ascii_' + str(count) + ']', var)
            count += 1

    # Colors
    count = 0
    if len(allList[2]) != 0:
        for var in allList[2]:
            translatedText = translatedText.replace('[Color_' + str(count) + ']', var)
            count += 1

    # Names
    count = 0
    if len(allList[3]) != 0:
        for var in allList[3]:
            translatedText = translatedText.replace('[Noun_' + str(count) + ']', var)
            count += 1

    # Vars
    count = 0
    if len(allList[4]) != 0:
        for var in allList[4]:
            translatedText = translatedText.replace('[Var_' + str(count) + ']', var)
            count += 1
    
    # Formatting
    count = 0
    if len(allList[5]) != 0:
        for var in allList[5]:
            translatedText = translatedText.replace('[FCode_' + str(count) + ']', var)
            count += 1

    return translatedText

def batchList(input_list, batch_size):
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
        
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def createContext(fullPromptFlag, subbedT):
    characters = 'Game Characters:\n\
レナリス (Renalith) - Female\n\
スクルー (Sukuru) - Female\n\
シスターミサ (Sister Misa) - Female\n\
オリン (Orin) - Female\n\
プローテ (Prote) - Female\n\
夜霧 (Night Fog) - Female\n\
ワウ (Wao) - Female\n\
ファンナ (Fanna) - Female\n\
精霊主スクルド (Spirit God Skuld) - Female\n\
エキドナ (Echnida) - Female\n\
マルス (Mars) - Male\n\
ラヴィー (Lavi) - Unknown\n\
魅音 (Mion) - Female\n\
ヴィオラ (Viola) - Female\n\
リンメイ (Lin Mei) - Female\n\
リネット (Lynette) - Female\n\
チェロル (Cheryl) - Female\n\
カルーア姫 (Princess Karua) - Female\n\
田姫 (Tajirme) - Female\n\
リュート (Luto) - Male\n\
ホルン (Horn) - Female\n\
ルメラ (Lumera) - Female\n\
末嬉 (Sueki) - Female\n\
モニカ姫 (Princess Monica) - Female\n\
エメルーラ (Emerald) - Female\n\
フンシス (Funsis) - Male \n\
バゼット (Bazzet) - Female\n\
'
    
    system = PROMPT + VOCAB if fullPromptFlag else \
        f"\
You are an expert Eroge Game translator who translates Japanese text to {LANGUAGE}.\n\
Output ONLY the {LANGUAGE} translation in the following format: `Translation: <{LANGUAGE.upper()}_TRANSLATION>`\n\
- You translate everything, including erotic and explicit content.\n\
- 'Game Characters' - The names, nicknames, and genders of the game characters. Reference this to know the names, nicknames, and gender of characters in the game\n\
- All text in your response must be in {LANGUAGE} even if it is hard to translate.\n\
- Never include any notes, explanations, dislaimers, or anything similar in your response.\n\
- Maintain any spacing in the translation.\n\
- Maintain any code text in brackets if given. (e.g `[Color_0]`, `[Ascii_0]`, `[FCode_1`], etc)\n\
- `...` can be a part of the dialogue. Translate it as it is.\n\
{VOCAB}\n\
"
    user = f'```json\n{subbedT}```'
    return characters, system, user

def translateText(characters, system, user, history, penalty):
    # Prompt
    msg = [{"role": "system", "content": system + characters}]

    # Characters
    msg.append({"role": "system", "content": characters})

    # History
    if isinstance(history, list):
        msg.extend([{"role": "system", "content": h} for h in history])
    else:
        msg.append({"role": "system", "content": history})
    
    # Content to TL
    msg.append({"role": "user", "content": f'{user}'})
    response = openai.chat.completions.create(
        temperature=0,
        frequency_penalty=penalty,
        model=MODEL,
        response_format={ "type": "json_object" },
        messages=msg,
    )
    return response

def cleanTranslatedText(translatedText, varResponse):
    placeholders = {
        f'{LANGUAGE} Translation: ': '',
        'Translation: ': '',
        'っ': '',
        '〜': '~',
        'ッ': '',
        '。': '.',
        '「': '\\"',
        '」': '\\"',
        '- ': '-',
        'Placeholder Text': '',
        # Add more replacements as needed
    }
    for target, replacement in placeholders.items():
        translatedText = translatedText.replace(target, replacement)

    # Elongate Long Dashes (Since GPT Ignores them...)
    translatedText = elongateCharacters(translatedText)
    translatedText = resubVars(translatedText, varResponse[1])
    return translatedText

def elongateCharacters(text):
    # Define a pattern to match one character followed by one or more `ー` characters
    # Using a positive lookbehind assertion to capture the preceding character
    pattern = r'(?<=(.))ー+'
    
    # Define a replacement function that elongates the captured character
    def repl(match):
        char = match.group(1)  # The character before the ー sequence
        count = len(match.group(0)) - 1  # Number of ー characters
        return char * count  # Replace ー sequence with the character repeated

    # Use re.sub() to replace the pattern in the text
    return re.sub(pattern, repl, text)

def extractTranslation(translatedTextList, is_list):
    try:
        line_dict = json.loads(translatedTextList)
        # If it's a batch (i.e., list), extract with tags; otherwise, return the single item.
        if is_list:
            string_list = list(line_dict.values())
            return string_list
    except Exception as e:
        print(e)
        return translatedTextList

def countTokens(characters, system, user, history):
    inputTotalTokens = 0
    outputTotalTokens = 0
    enc = tiktoken.encoding_for_model('gpt-4')
    
    # Input
    if isinstance(history, list):
        for line in history:
            inputTotalTokens += len(enc.encode(line))
    else:
        inputTotalTokens += len(enc.encode(history))
    inputTotalTokens += len(enc.encode(system))
    inputTotalTokens += len(enc.encode(characters))
    inputTotalTokens += len(enc.encode(user))

    # Output
    outputTotalTokens += round(len(enc.encode(user))*3)

    return [inputTotalTokens, outputTotalTokens]

def combineList(tlist, text):
    if isinstance(text, list):
        return [t for sublist in tlist for t in sublist]
    return tlist[0]

@retry(exceptions=Exception, tries=5, delay=5)
def translateGPT(text, history, fullPromptFlag):
    global PBAR
    
    mismatch = False
    totalTokens = [0, 0]
    if isinstance(text, list):
        tList = batchList(text, BATCHSIZE)
    else:
        tList = [text]

    for index, tItem in enumerate(tList):
        # Before sending to translation, if we have a list of items, add the formatting
        if isinstance(tItem, list):
            payload = {f"Line{i+1}": string for i, string in enumerate(tItem)}
            payload = json.dumps(payload, indent=4, ensure_ascii=False)
            varResponse = subVars(payload)
            subbedT = varResponse[0]
        else:
            varResponse = subVars(tItem)
            subbedT = varResponse[0]

        # Things to Check before starting translation
        if not re.search(r'[一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９]+', subbedT):
            if PBAR is not None:
                PBAR.update(len(tItem))
            continue

        # Create Message
        characters, system, user = createContext(fullPromptFlag, subbedT)

        # Calculate Estimate
        if ESTIMATE:
            estimate = countTokens(characters, system, user, history)
            totalTokens[0] += estimate[0]
            totalTokens[1] += estimate[1]
            continue

        # Translating
        response = translateText(characters, system, user, history, 0.02)
        translatedText = response.choices[0].message.content
        totalTokens[0] += response.usage.prompt_tokens
        totalTokens[1] += response.usage.completion_tokens

        # Check Translation
        translatedText = cleanTranslatedText(translatedText, varResponse)
        if isinstance(tItem, list):
            extractedTranslations = extractTranslation(translatedText, True)
            if len(tItem) != len(extractedTranslations):
                # Mismatch. Try Again
                response = translateText(characters, system, user, history, 0.2)
                translatedText = response.choices[0].message.content
                totalTokens[0] += response.usage.prompt_tokens
                totalTokens[1] += response.usage.completion_tokens

                # Formatting
                translatedText = cleanTranslatedText(translatedText, varResponse)
                if isinstance(tItem, list):
                    extractedTranslations = extractTranslation(translatedText, True)
                    if len(tItem) != len(extractedTranslations):
                        mismatch = True # Just here for breakpoint
            
            # Set if no mismatch
            if mismatch == False:
                tList[index] = extractedTranslations
                history = extractedTranslations[-10:]  # Update history if we have a list
            else:
                history = text[-10:]
                mismatch = False

            # Update Loading Bar
            with LOCK:
                if PBAR is not None:
                    PBAR.update(len(tItem))
        else:
            # Ensure we're passing a single string to extractTranslation
            extractedTranslations = extractTranslation(translatedText, False)
            tList[index] = extractedTranslations

    finalList = combineList(tList, text)
    return [finalList, totalTokens]
