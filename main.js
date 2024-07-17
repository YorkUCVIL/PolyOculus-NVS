

var indist_scenes = [
	'b325dadd4c13e439',
	'ccc439d4b28c87b2',
	'007b4ae7c05f2ea2',
	'80f5dfcec1498115',
	'000db54a47bd43fe',
	'b11afd8c64079de8',
];
var dfm_scenes = [
	'b325dadd4c13e439-128',
	'ccc439d4b28c87b2-128',
	'007b4ae7c05f2ea2-128',
	'80f5dfcec1498115-128',
	'000db54a47bd43fe-128',
	'b11afd8c64079de8-128',
];
var mp3d_scenes = [
	'EU6Fwq7SyZv-02',
	'EU6Fwq7SyZv-51',
	'QUCTc6BB5sX-25',
	'TbHJrupSAjP-09',
	'oLBMNvg9in8-65',
	'zsNo4HB9uLZ-05',
];
var spin_scenes = [
	'061f829d3dd2e46e',
	'15f8a54e4822f355',
	'9abc9859c773ff50',
	'6bc457ceb8a81ed5',
];
var stereo_scenes = [
	'000c3ab189999a83'
]

generation_sets = [[47, 46, 49, 66, 30], [45, 27, 63, 25], [50, 32, 10], [61, 43, 23], [59, 41, 4], [40, 57, 39, 20], [13, 52, 69], [71, 54, 35], [55, 16, 73], [56, 38, 75], [1, 7], [19], [48], [42], [53], [44], [51], [21], [37], [64], [68], [34], [58], [74], [26], [29], [28], [31], [65], [67], [24], [33], [62], [70], [22], [36], [60], [72], [2], [18], [9], [11], [6], [14], [8], [12], [5], [15], [3], [17]];
generation_sets_conditioning = [[0], [46, 47], [49, 30, 27], [63, 45, 25], [61, 43, 23, 25], [41, 59, 23], [32, 30, 50, 49], [69, 52, 50, 32], [54, 35, 13, 71], [55, 54, 73], [20, 4, 25, 27], [38, 16], [49, 0, 47], [43, 41, 23], [52, 54, 35], [45, 43, 42], [50, 52, 53], [20, 23, 40], [38, 35, 55], [63, 66, 46], [69, 66, 49], [35, 32, 52], [59, 57, 40], [73, 75, 55], [25, 27, 7], [30, 27, 0], [27, 29, 30], [30, 32, 29], [64, 66, 63], [68, 66, 69], [25, 23, 26], [34, 32, 35], [61, 63, 64], [71, 69, 68], [23, 21, 24], [35, 37, 34], [59, 61, 58], [73, 71, 74], [1, 4, 21], [19, 16, 37], [10, 7, 28], [10, 13, 9], [7, 4, 25], [13, 16, 33], [7, 9, 6], [13, 11, 14], [4, 6, 7], [16, 14, 13], [2, 4, 1], [18, 16, 19]];

var methods = ['sde','set-markov','set-all','set-keyframed'];
var spin_methods = ['set-markov','set-all','set-keyframed'];
var dfm_methods = ['dfm-1','dfm-2','set-keyframed'];
var stereo_methods = ['set-markov','set-keyframed'];
class Sample_viewer{
	/*
	Viewer setup needs a mix of HTML and JS
	See the HTML and this class to see how to structure the HTML elements, their ids, and JS callbacks
	The prefix argument i used to identify the viewer, needs to be consistent with HTML for the JS to find the right elements
	*/
	constructor(prefix,max_idx,scene_codes,variants,methods){
		this.methods = methods;
		this.variants = variants;
		this.scene_codes = scene_codes
		this.n_scenes = scene_codes.length;
		this.prefix = prefix;
		this.max_idx = max_idx;
		this.cur_frame = 0;
		this.cur_sample = 0;
		this.variant = variants[0];
		this.scene_code = scene_codes[0];
		this.need_stop_anim = false;
		this.interval_id = null;
		this.anim_dir = 1;
		for (let i=0;i<this.n_scenes;i++){
			//document.getElementById(`${this.prefix}-scene-selector`).innerHTML += `<div onclick="${this.prefix}_viewer.change_scene(\'${scene_codes[i]}\');" class="col-1"> <img style="border-radius:1em;" class=selectable src="assets/individual-frames/${variants[0]}/set-markov/${scene_codes[i]}/samples/0000/images/0000.webp"> </div>`;
			document.getElementById(`${this.prefix}-scene-selector`).innerHTML += `<div onclick="${this.prefix}_viewer.change_scene(\'${scene_codes[i]}\');" class="col-1"> <img style="border-radius:1em;" class=selectable src="assets/init-ims/${scene_codes[i]}.webp"> </div>`;
		}
	}
	update_ims(){
		/*
		This is the main method that takes all the image parameters and updates the images in the web page
		*/
		for (let method of this.methods){
			if (this.cur_frame == 0){
				// This is a hack used by my project to reduce the size of the supplemental material
				// In this project the first frame is always the same so this code just reuses the same image
				document.getElementById(`${this.prefix}-${method}`).src = `assets/init-ims/${this.scene_code}.webp`;
			}else{
				let frame_padded = this.cur_frame.toString().padStart(4,0);
				let sample_padded = this.cur_sample.toString().padStart(4,0);
				document.getElementById(`${this.prefix}-${method}`).src = `assets/individual-frames/${this.variant}/${method}/${this.scene_code}/samples/${sample_padded}/images/${frame_padded}.webp`;
			}
		}
	}

	/* ===================================================================================
	The methods below are used for image control, called by pushing buttons on the HTML
	=================================================================================== */
	change_scene(scene_code){
		this.scene_code = scene_code;
		this.update_ims();
	}
	change_variant(name){
		this.variant = name;
		if (this.variants){
			for (let nn of this.variants){
				document.getElementById(`${nn}_selector`).style.backgroundColor = '';
				document.getElementById(`${nn}_selector`).style.borderRadius = '1em';
			}
			document.getElementById(`${name}_selector`).style.backgroundColor = 'lightgrey';
			document.getElementById(`${name}_selector`).style.borderRadius = '1em';
		}
		this.update_ims();
	}
	change_sample(idx){
		this.cur_sample = idx;
		this.update_ims();
		for (let i=0;i<3;i++){
			document.getElementById(`${this.prefix}_sample_selector_${i+1}`).style.backgroundColor = 'rgb(240,240,240)';
		}
		document.getElementById(`${this.prefix}_sample_selector_${idx+1}`).style.backgroundColor = 'lightgrey';
	}


	/* ===================================================================================
	The methods below are used for automatic playback
	=================================================================================== */
	change_frame(idx){
		/*
		This is called when the user clicks and drags on the slider to see a specific frame
		This also stops the automatic playback
		*/
		this.stop_anim();
		this.cur_frame = parseInt(idx);
		this.update_ims();
	}
	next_frame(){
		/*
		This is used internally to play the sequence forward and backward, and also moves the slider to show the user what frame is being shown
		*/
		this.cur_frame += this.anim_dir;
		if (this.cur_frame >= this.max_idx) {this.anim_dir=-1;}
		if (this.cur_frame <= 0) {this.anim_dir=1;}
		document.getElementById(`${this.prefix}_frame_control`).value = this.cur_frame;
		this.update_ims();
	}
	cycle_frames(delay){
		/*
		Starts the automatic playback using JS intervals, see next_frame() to see the cycling behavior
		*/
		this.stop_anim();
		this.interval_id = setInterval(function() {
			this.next_frame();
		}.bind(this), delay);
		this.update_ims();
	}
	stop_anim(){
		if (this.interval_id){clearInterval(this.interval_id);}
		this.interval_id = null;
	}
};

class Sample_viewer_cyclic extends Sample_viewer {
	next_frame(){
		/*
		This is used internally to play the sequence forward and backward, and also moves the slider to show the user what frame is being shown
		*/
		this.cur_frame += this.anim_dir;
		if (this.cur_frame > this.max_idx){
			this.cur_frame = 0;
		}
		document.getElementById(`${this.prefix}_frame_control`).value = this.cur_frame;
		this.update_ims();
	}

	first_last(){
		if (this.cur_frame == this.max_idx){
			this.cur_frame = 0;
		}
		else{
			this.cur_frame = this.max_idx;
		}
		document.getElementById(`${this.prefix}_frame_control`).value = this.cur_frame;
		this.update_ims();
	}

	cycle_first_last(){
		this.stop_anim();
		this.interval_id = setInterval(function() {
			this.first_last();
		}.bind(this), 400);
		this.update_ims();
	}
};

class Sample_viewer_stereo extends Sample_viewer {
	constructor(prefix,max_idx,scene_codes,variants){
		super(prefix,max_idx,scene_codes,variants);
		this.lr_flipped = false;
		this.nice_names = {
			'set-keyframed':'Group Sampling',
			'set-markov':'Naive Markov Sampling',
		};
	}

	update_ims(){
		/*
		This is the main method that takes all the image parameters and updates the images in the web page
		*/
		for (let method of stereo_methods){
			//if (this.cur_frame == 0){
			//    // This is a hack used by my project to reduce the size of the supplemental material
			//    // In this project the first frame is always the same so this code just reuses the same image
			//    document.getElementById(`${this.prefix}-${method}`).src = `assets/individual-frames/initial-frames/${this.prefix}/${this.base_im}.webp`;
			//}else{
				if (this.lr_flipped){
					var frame_padded_left = (this.cur_frame*2).toString().padStart(4,0);
					var frame_padded_right = (this.cur_frame*2+1).toString().padStart(4,0);
					document.getElementById(`${this.prefix}-${method}-left-label`).innerHTML = `${this.nice_names[method]} - RIGHT VIEW`;
					document.getElementById(`${this.prefix}-${method}-right-label`).innerHTML = `${this.nice_names[method]} - LEFT VIEW`;
				}
				else{
					var frame_padded_left = (this.cur_frame*2+1).toString().padStart(4,0);
					var frame_padded_right = (this.cur_frame*2).toString().padStart(4,0);
					document.getElementById(`${this.prefix}-${method}-left-label`).innerHTML = `${this.nice_names[method]} - LEFT VIEW`;
					document.getElementById(`${this.prefix}-${method}-right-label`).innerHTML = `${this.nice_names[method]} - RIGHT VIEW`;
				}
				let sample_padded = this.cur_sample.toString().padStart(4,0);
				document.getElementById(`${this.prefix}-${method}-left`).src = `assets/individual-frames/${this.variant}/${method}/${this.scene_code}/samples/${sample_padded}/images/${frame_padded_left}.webp`;
				document.getElementById(`${this.prefix}-${method}-right`).src = `assets/individual-frames/${this.variant}/${method}/${this.scene_code}/samples/${sample_padded}/images/${frame_padded_right}.webp`;
			//}
		}
	}
	flip_ims(){
		if (this.lr_flipped){
			this.lr_flipped = false;
		}
		else{
			this.lr_flipped = true;
		}
		this.update_ims();
	}
};

class Cloud_viewer{
	/*
	Viewer setup needs a mix of HTML and JS
	See the HTML and this class to see how to structure the HTML elements, their ids, and JS callbacks
	The prefix argument i used to identify the viewer, needs to be consistent with HTML for the JS to find the right elements
	*/
	constructor(prefix,n_vert,n_hor,scene_code){
		this.scene_code = scene_code;
		this.prefix = prefix;
		this.n_vert = n_vert;
		this.n_hor = n_hor;
		this.visualizer_container = document.getElementById(`${prefix}_visualizer`);
		this.selector_container = document.getElementById(`${prefix}_selector`);
		this.create_visualizer();
		this.create_selector();
	}

	change_view(idx){
		var frame_padded = idx.toString().padStart(4,0);
		document.getElementById(`${this.prefix}-viewer`).src = `assets/individual-frames/cloud/set-keyframed/${this.scene_code}/samples/0000/images/${frame_padded}.webp`;
	}

	create_selector(){
		let counter = 1;
		let mid_r = Math.floor(this.n_vert/2);
		let mid_c = Math.floor(this.n_hor/2);
		for (let r=0;r<this.n_vert;r++){
			let row = document.createElement('div');
			row.style.width = '100%';
			row.style.display = 'flex';
			row.style.marginBottom = '0.1em';
			//row.style.height = '4vw';
			this.selector_container.appendChild(row);
			for (let c=0;c<this.n_hor;c++){
				let selector = document.createElement('div');
				selector.style.borderRadius = '0.5em';
				selector.style.width = '5%';
				selector.style.aspectRatio = '1';
				selector.style.marginLeft = 'auto';
				selector.style.marginRight = 'auto';
				selector.style.display = 'inline-block';
				selector.style.backgroundColor = '#919191';
				let idx = -1;
				if (r==mid_r && c==mid_c){
					idx = 0;
					selector.id = `${this.prefix}-sel-0`;
					selector.style.borderStyle = 'solid';
					selector.style.borderWidth = '0.3em';
					selector.style.borderColor = 'red';
					selector.style.backgroundColor = 'green';
				}
				else{
					idx = counter;
					selector.id = `${this.prefix}-sel-${counter}`;
					counter++;
				}
				let n_poses = this.n_hor * this.n_vert;
				selector.addEventListener('mouseover',function(e){
					for (let i=0;i<n_poses;i++){ // reset colour of everything
						document.getElementById(`${this.prefix}-sel-${i}`).style.backgroundColor = '#919191';
					}
					selector.style.backgroundColor = 'green';
					this.change_view(idx);
				}.bind(this));
				row.appendChild(selector);
			}
		}
	}

	create_visualizer(){
		let counter = 1;
		let mid_r = Math.floor(this.n_vert/2);
		let mid_c = Math.floor(this.n_hor/2);
		for (let r=0;r<this.n_vert;r++){
			let row = document.createElement('div');
			row.style.width = '100%';
			row.style.display = 'flex';
			row.style.marginBottom = '0.1em';
			this.visualizer_container.appendChild(row);
			for (let c=0;c<this.n_hor;c++){
				let selector = document.createElement('div');
				selector.style.borderRadius = '0.5em';
				selector.style.aspectRatio = '1';
				selector.style.width = '5%';
				selector.style.marginLeft = 'auto';
				selector.style.marginRight = 'auto';
				selector.style.display = 'inline-block';
				selector.style.backgroundColor = '#919191';
				if (r==mid_r && c==mid_c){
					selector.id = `${this.prefix}-0`;
					selector.style.borderStyle = 'solid';
					selector.style.borderWidth = '0.3em';
					selector.style.borderColor = 'red';
				}
				else{
					selector.id = `${this.prefix}-${counter}`;
					counter++;
				}
				row.appendChild(selector);
			}
			//this.visualizer_container.appendChild(document.createElement('br'));
		}
	}

	get_already_generated(idx){
		let generated = [0]; // 0 is given
		for (let i=0;i<idx;i++){
			for (let j of generation_sets[i]){
				generated.push(j);
			}
		}
		return generated;
	}
	
	change_step(idx){
		let n_poses = this.n_hor * this.n_vert;
		let gen = generation_sets[idx];
		let cond = generation_sets_conditioning[idx];
		let generated = this.get_already_generated(idx);
		for (let i=0;i<n_poses;i++){ // reset colour of everything
			document.getElementById(`${this.prefix}-${i}`).style.backgroundColor = '#919191';
		}
		for (let i of generated){ // generated
			document.getElementById(`${this.prefix}-${i}`).style.backgroundColor = 'green';
		}
		for (let i of gen){ // generated
			document.getElementById(`${this.prefix}-${i}`).style.backgroundColor = 'orange';
		}
		for (let i of cond){ // conditioned
			document.getElementById(`${this.prefix}-${i}`).style.backgroundColor = '#9191ff';
		}
	}
};

// create the viewer here to make it global, and accessible from the HTML
var indist_viewer = null;
var mp3d_viewer = null;
var dfm_viewer = null;
var spin_viewer = null;
var stereo_viewer = null;
var cloud_viewer = null;

document.addEventListener("DOMContentLoaded", function() {
	// create the viewer, and set the initial frame
	indist_viewer = new Sample_viewer('indist',20,indist_scenes,['indist'],methods);
	indist_viewer.change_frame(0);
	indist_viewer.change_sample(0);
	indist_viewer.change_variant('indist');

	mp3d_viewer = new Sample_viewer('mp3d',20,mp3d_scenes,['mp3d'],methods);
	mp3d_viewer.change_frame(0);
	mp3d_viewer.change_sample(0);
	mp3d_viewer.change_variant('mp3d');

	dfm_viewer = new Sample_viewer('dfm',20,dfm_scenes,['dfm'],dfm_methods);
	dfm_viewer.change_frame(0);
	dfm_viewer.change_sample(0);
	dfm_viewer.change_variant('dfm');

	spin_viewer = new Sample_viewer_cyclic('spin',9,spin_scenes,['spin'],spin_methods);
	spin_viewer.change_frame(0);
	spin_viewer.change_sample(0);
	spin_viewer.change_variant('spin');

	stereo_viewer = new Sample_viewer_stereo('stereo',20,stereo_scenes,['stereo']);
	stereo_viewer.change_frame(0);
	stereo_viewer.change_sample(0);
	stereo_viewer.change_variant('stereo');

	cloud_viewer = new Cloud_viewer('cloud',4,19,'6bc457ceb8a81ed5');
	cloud_viewer.change_step(0);
	cloud_viewer.change_view(0);
});
