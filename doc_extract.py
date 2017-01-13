import os
import sys
import docx

if __name__ == "__main__":
    sys.path.extend(['C:\\Users\\heyyj\\PycharmProjects\\Schizotypy'])
    #number of docx files in folder
    num_docs = 67
    info_dir = 'data\\test'
    count = 0
    for filename in os.listdir(info_dir):
        assert filename.endswith(".docx")
        print count
        print filename
        extended_name = 'data\\test\\' + filename
        doc = docx.Document(extended_name)

        #create file for only patient responses
        text_name_patient = 'data\\IPII_patient\\' + filename.rstrip('.docx') + '_patient.txt'
        f_patient = open(text_name_patient, 'w')

        #create file for pairs of data
        text_name_pair = 'data\\IPII_pairs\\' + filename.rstrip('.docx') + '_pair.txt'
        f_pair = open(text_name_pair, 'w')

        prev = None

        for para in doc.paragraphs:
            if para.text:
                lower_str = para.text.lower().encode('ascii', 'ignore' )
                cleaned_str = lower_str.lstrip('male voice: ')
                cleaned_str = cleaned_str.lstrip('female voice: ')
                cleaned_str = cleaned_str.lstrip('s: ')

                #check for new patient response
                if lower_str.startswith('male voice:') or lower_str.startswith('female voice:') or lower_str.startswith('s:'):

                    f_pair.write('\n')
                    f_pair.write('v: ' + cleaned_str)
                    f_patient.write('\n')
                    f_patient.write(cleaned_str)
                    prev = 'patient'

                #check for new interviewer response
                elif lower_str.startswith('interviewer:') or para.runs[0].italic == True:
                    f_pair.write('\n')
                    f_pair.write('i : ' + cleaned_str)
                    prev = 'interviewer'

                #check for continued answers in new paragraph
                else:
                    if prev == 'patient':
                        f_patient.write(cleaned_str)
                        f_pair.write('v: ' + lower_str)

                    elif prev == 'interviewer':
                        f_pair.write('i: ' + lower_str)

        f_patient.close()
        f_pair.close()

        count+=1

    if count == num_docs:
        print 'Extraction Complete!'
