"""
Convert msp files

Writes three files: mgf with the spectra; PEPREC with the peptide sequences; meta with additional metainformation.
Arguments:
	arg1 path to msp file
	arg2 TITLE
"""
#CHaNGED MODS!!

import sys

AminoMass = {}
AminoMass['A']= 71.037114
AminoMass['C']= 103.009185
AminoMass['D']= 115.026943
AminoMass['E']= 129.042593
AminoMass['F']= 147.068414
AminoMass['G']= 57.021464
AminoMass['H']= 137.058912
AminoMass['I']= 113.084064
AminoMass['K']= 128.094963
AminoMass['L']= 113.084064
AminoMass['M']= 131.040485
AminoMass['N']= 114.042927
AminoMass['P']= 97.052764
AminoMass['Q']= 128.058578
AminoMass['R']= 156.101111
AminoMass['S']= 87.032028
AminoMass['T']= 101.047679
AminoMass['V']= 99.068414
AminoMass['W']= 186.079313
AminoMass['Y']= 163.063329

fpip = open(sys.argv[1]+'.PEPREC','w')
fpip.write("spec_id modifications peptide\n")
fmgf = open(sys.argv[1]+'.PEPREC.mgf','w')
fmeta = open(sys.argv[1]+'.PEPREC.meta','w')

PTMs = {}

specid = 1
with open(sys.argv[1]) as f:
	peptide = None
	charge = None
	parentmz = None
	mods = None
	purity = None
	HCDenergy = None
	read_spec = False
	mgf = ""
	prev = 'A'
	#sys.stderr.write(prev)
	for row in f:
		if read_spec:
			l = row.rstrip().split('\t')
			if len(l) != 3:
				if peptide[0] != prev:
					prev = peptide[0]
					#sys.stderr.write(prev)

				tmp = mods.split('/')
				if tmp[0] != '0':
					m = ""
					for i in range(1,len(tmp)):
						tmp2=tmp[i].split(',')
						if (tmp2[0]=='0') & (tmp2[2]=='iTRAQ'):
							m += '0|'+tmp2[2] +'|'
						else:
							#m += str(int(tmp2[0])+1)+'|'+tmp2[2] +'|'
							m += str(int(tmp2[0])+1)+'|'+tmp2[2] + peptide[int(tmp2[0])] + '|'
						if not tmp2[2] in PTMs: PTMs[tmp2[2]] = 0
						PTMs[tmp2[2]] += 1
					fpip.write('%s%i %s %s\n'%(sys.argv[2],specid,m[:-1],peptide))
				else:
					fpip.write('%s%i  %s\n'%(sys.argv[2],specid,peptide))

				fmeta.write('%s%i %s %s %s %s %s\n'%(sys.argv[2],specid,charge,peptide,parentmz,purity,HCDenergy))

				# THIS IS NOT A PROBLEM: MW is nothing
				if tmp[0] == '0':
					if 'X' in peptide: continue
					tmp1 = 18.010601 + sum([AminoMass[x] for x in peptide])
					tmp2 = (float(parentmz) * (float(charge))) - ((float(charge))*1.007825035) #or 0.0073??
					if abs(tmp1-tmp2) > 0.5:
						print "."


				buf = "BEGIN IONS\n"
				buf += "TITLE="+sys.argv[2]+str(specid) + '\n'
				buf += "CHARGE="+str(charge) + '\n'
				buf += "PEPMASS="+parentmz + '\n'
				fmgf.write(buf+mgf+"END IONS"+'\n')

				specid += 1
				read_spec = False
				mgf = ""
				continue
			else:
				tt = float(l[1])
				mgf += ' '.join([l[0],l[1]]) + '\n'
				continue
		if row.startswith("Name:"):
			l = row.rstrip().split(' ')
			tmp = l[1].split('/')
			peptide = tmp[0].replace('(O)','')
			charge = tmp[1].split('_')[0]
			continue
		if row.startswith("Comment:"):
			l = row.rstrip().split(' ')
			for i in range(1,len(l)):
				if l[i].startswith("Mods="):
					tmp = l[i].split('=')
					mods = tmp[1]
				if l[i].startswith("Parent="):
					tmp = l[i].split('=')
					parentmz = tmp[1]
				if l[i].startswith("Purity="):
					tmp = l[i].split('=')
					purity = tmp[1]
				if l[i].startswith("HCD="):
					tmp = l[i].split('=')
					HCDenergy = tmp[1].replace('eV','')
			continue
		if row.startswith("Num peaks:"):
			read_spec = True
			continue

fmgf.close()
fpip.close()
fmeta.close()

print PTMs
