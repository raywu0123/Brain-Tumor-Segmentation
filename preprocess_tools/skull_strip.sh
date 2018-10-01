bse_dir=$1
echo "bse_dir: " $bse_dir

result_dir=$2
echo "result_dir:" $result_dir
mkdir -p $result_dir
mkdir -p "$result_dir/image"
mkdir -p "$result_dir/mask"

for filepath in /shares/Public/challenge.0924/image/*; do
	filename=$(basename "$filepath" "")	
	echo $filename
	$bse_dir -i $filepath \
		 -o "$result_dir/image/$filename" \
		 --mask "$result_dir/mask/$filename" \
		 --norotate
done
