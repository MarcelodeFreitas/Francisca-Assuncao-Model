Notebook com o fluxo completo para a geração de máscaras para o Bolbo Olfativo.

1 - O input são scans (.nii.gz) resampled através do FSL para uma pixdim de (0.5,0.5,1).
Nota: Os scans da pasta input já sofreram o resampling do passo 1.

2 - O segundo passo (célula 7) é a utilização do model de classificação de slices em dois grupos, com OB e sem OB.

3 - De seguida, utiliza-se o modelo de regressão para a descoberta do centro de massa das slices com OB para, posteriormente, ser feita a Bounding Box (célula 8). A bounding box, neste caso, tem a dimensão fixa de 32x32.

4 - Segue-se a segmentação do OB (célula 10).

5 - Por fim, geram-se as máscaras que surgiram da rede de segmentação com as dimensões iniciais de (174,190,30) (última célula). As máscaras são geradas scan a scan. É necessário colocar o número do scan (num_scan=X).


Nota:

-- > Para o Resample do passo 1:

#! /bin/sh
for filename in *.nii.gz ; do 
	fname=`$FSLDIR/bin/remove_ext ${filename}`
	/usr/local/fsl/bin/fslcreatehd 240 240 101 1 0.5 0.5 1 1 0 0 0 4  /path/${fname}_tmp.nii.gz ; /usr/local/fsl/bin/flirt -in /path/${fname}.nii.gz -applyxfm -init /usr/local/fsl/etc/flirtsch/ident.mat -out /path/${fname}_new.nii.gz -paddingsize 0.0 -interp trilinear -ref /path/${fname}_tmp
done

Onde (240,240,101) é a shape original e o (0.5, 0.5, 1) é a pixdim pedida.